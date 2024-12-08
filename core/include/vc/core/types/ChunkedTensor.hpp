#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "z5/multiarray/xtensor_access.hxx"

#include <random>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

struct vec3i_hash {
    size_t operator()(cv::Vec3i p) const
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

struct passTroughComputor
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    const std::string UNIQUE_ID_STRING = "";
    template <typename T, typename E> void compute(const T &large, T &small)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

static uint64_t miss = 0;
static uint64_t total = 0;
static uint64_t chunk_compute_collisions = 0;
static uint64_t chunk_compute_total = 0;

template <typename T, typename C> class Chunked3dAccessor;

static std::string tmp_name_proc_thread()
{
    std::stringstream ss;
    ss << "tmp_" << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
    };
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache, const fs::path &cache_root) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
        
        if (cache_root.empty())
            return;

        if (!_compute_f.UNIQUE_ID_STRING.size())
            throw std::runtime_error("requested fs cache for compute function without identifier");        
        
        fs::path root = cache_root/_compute_f.UNIQUE_ID_STRING;
        
        fs::create_directories(root);
        
        //create cache dir while others are competing to do the same
        for(int r=0;r<1000 && _cache_dir.empty();r++) {
            std::set<std::string> paths;
            for (auto const& entry : fs::directory_iterator(root))
                if (fs::is_directory(entry) && fs::exists(entry.path()/"meta.json") && fs::is_regular_file(entry.path()/"meta.json")) {
                    paths.insert(entry.path());
                    std::ifstream meta_f(entry.path()/"meta.json");
                    nlohmann::json meta = nlohmann::json::parse(meta_f);
                    fs::path src = fs::canonical(meta["dataset_source_path"]);
                    if (src == fs::canonical(ds->path())) {
                        _cache_dir = entry.path();
                        break;
                    }
                }
                
            if (!_cache_dir.empty())
                continue;
            
            //try generating our own cache dir atomically
            fs::path tmp_dir = cache_root/tmp_name_proc_thread();
            fs::create_directories(tmp_dir);
            
            nlohmann::json meta;
            meta["dataset_source_path"] = fs::canonical(ds->path()).string();
            std::ofstream o(tmp_dir/"meta.json");
            o << std::setw(4) << meta << std::endl;
            
            fs::path tgt_path;
            for(int i=0;i<1000;i++) {
                tgt_path = root/std::to_string(i);
                if (paths.count(tgt_path.string()))
                    continue;
                try {
                    fs::rename(tmp_dir, tgt_path);
                }
                catch (fs::filesystem_error){
                    continue;
                }
                _cache_dir = tgt_path;
                break;
            }
        }
        
        if (_cache_dir.empty())
            throw std::runtime_error("could not create cache dir - maybe too many caches in cache root (max 1000!)");
        
    };
    size_t calc_off(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        return p[0] + p[1]*s + p[2]*s*s;
    }
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);

        return _chunks[id][calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        T *chunk = nullptr;

        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk[calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    fs::path id_path(const fs::path &dir, const cv::Vec3i &id)
    {
        return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
    }

    T *cache_chunk_safe_mmap(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;

        fs::path tgt_path = id_path(_cache_dir, id);
        size_t len = s*s*s;

        if (fs::exists(tgt_path)) {
            int fd = open(tgt_path.string().c_str(), O_RDWR);
            T *chunk = (T*)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            _mutex.lock();
            if (!_chunks.count(id)) {
                _chunks[id] = chunk;
            }
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                munmap(chunk, len);
                chunk = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

            return chunk;
        }

        fs::path tmp_path;
        _mutex.lock();
        std::stringstream ss;
        ss << this << "_" << std::this_thread::get_id() << "_" << _tmp_counter++;
        tmp_path = fs::path(_cache_dir) / ss.str();
        _mutex.unlock();
        int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        ftruncate(fd, len);
        T *chunk = (T*)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);


        CHUNKT small = xt::empty<T>({s,s,s});
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

        readArea3D(large, offset, _ds, _cache);

        _compute_f.template compute<CHUNKT,T>(large, small);

        for(int i=0;i<len;i++)
            chunk[i] = (&small(0,0,0))[i];

        _mutex.lock();
        if (!_chunks.count(id)) {
            _chunks[id] = chunk;
            int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

            if (ret)
                throw std::runtime_error("oops rename failed!");
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            munmap(chunk, len);
            unlink(tmp_path.string().c_str());
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }


    T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);
        CHUNKT small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            _compute_f.template compute<CHUNKT,T>(large, small);

            T *chunk = nullptr;

            _mutex.lock();
            if (!_chunks.count(id)) {
                chunk = (T*)malloc(s*s*s);
                memcpy(chunk, &small(0,0,0), s*s*s);
                _chunks[id] = chunk;
            }
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                chunk = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

            return chunk;
    }

    // T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    // {
    //     auto s = C::CHUNK_SIZE;
    //     CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
    //     large = xt::full_like(large, C::FILL_V);
    //     CHUNKT small = xt::empty<T>({s,s,s});
    //
    //     cv::Vec3i offset =
    //     {id[0]*s-_border,
    //         id[1]*s-_border,
    //         id[2]*s-_border};
    //
    //     readArea3D(large, offset, _ds, _cache);
    //
    //     _compute_f.template compute<CHUNKT,T>(large, small);
    //
    //     _mutex.lock();
    //     if (!_chunks.count(id))
    //         _chunks[id] = small;
    //     else {
    //         #pragma omp atomic
    //         chunk_compute_collisions++;
    //         delete small;
    //         small = _chunks[id];
    //     }
    //     #pragma omp atomic
    //     chunk_compute_total++;
    //     _mutex.unlock();
    //
    //     return small;
    // }

    T *cache_chunk_safe(const cv::Vec3i &id)
    {
        if (_cache_dir.empty())
            return cache_chunk_safe_alloc(id);
        else
            return cache_chunk_safe_mmap(id);
    }

    T *cache_chunk(const cv::Vec3i &id) {
        return cache_chunk_safe(id);
        // auto s = C::CHUNK_SIZE;
        // CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        // large = xt::full_like(large, C::FILL_V);
        // CHUNKT *small = new CHUNKT();
        // *small = xt::empty<T>({s,s,s});
        //
        // cv::Vec3i offset =
        // {id[0]*s-_border,
        //     id[1]*s-_border,
        //     id[2]*s-_border};
        //
        //     readArea3D(large, offset, _ds, _cache);
        //
        //     _compute_f.template compute<CHUNKT,T>(large, *small);
        //
        //     _chunks[id] = small;
        //
        // return small;
    }

    T *chunk(const cv::Vec3i &id) {
        if (!_chunks.count(id))
            return cache_chunk(id);
        return _chunks[id];
    }

    T *chunk_safe(const cv::Vec3i &id) {
        T *chunk = nullptr;
        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk;
    }

    auto shape()
    {
        return _ds->shape();
    }

    std::unordered_map<cv::Vec3i,T*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
    uint64_t _tmp_counter = 0;
    fs::path _cache_dir;
};

void print_accessor_stats();

template <typename T, typename C>
class Chunked3dAccessor
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {};

    static Chunked3dAccessor &create(Chunked3d<T,C> &ar)
    {
        return Chunked3dAccessor(ar);
    }

    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk(p);
        }

        total++;

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }



    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }

    T& safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk_safe(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk_safe(p);
        }

        #pragma omp atomic
        total++;

        // size_t pos_xt = &_chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]) - &_chunk->operator()(0,0,0);
        // if (pos_xt != _ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})) {
        //     std::cout << pos_xt << cv::Vec3i({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]}) << _ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]}) << std::endl;
        //     throw std::runtime_error("fix calc_off!");
        // }

        // return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }

    T& safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    void get_chunk(const cv::Vec3i &p)
    {
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk(id);
        _corner = id*C::CHUNK_SIZE;
    }

    void get_chunk_safe(const cv::Vec3i &p)
    {
        #pragma omp atomic
        miss++;
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk_safe(id);
        _corner = id*C::CHUNK_SIZE;
    }

    Chunked3d<T,C> &_ar;
protected:
    T *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

//FIXME add thread safe variant!
template <typename T, typename C>
class CachedChunked3dInterpolator
{
public:
    CachedChunked3dInterpolator(Chunked3d<T,C> &t) : _a(t)
    {
        _shape = {t.shape()[0], t.shape()[1], t.shape()[2]};
    };

    Chunked3dAccessor<T,C> _a;

    template <typename V> void Evaluate(const V &z, const V &y, const V &x, V *out)
    {
        cv::Vec3d f = {val(z),val(y),val(x)};
        cv::Vec3i corner = {floor(f[0]),floor(f[1]),floor(f[2])};
        for(int i=0;i<3;i++) {
            corner[i] = std::max(corner[i], 0);
            corner[i] = std::min(corner[i], _shape[i]-2);
        }
        V fc[3] = { z - V(corner[0]), y - V(corner[1]), x - V(corner[2])};
        for(int i=0;i<3;i++) {
            if (fc[i] < V(0))
                fc[i] = V(0);
            else if (fc[i] > V(1))
                fc[i] = V(1);
        }

        V c000 = V(_a.safe_at(corner));
        V c100 = V(_a.safe_at(corner+cv::Vec3i(1,0,0)));
        V c010 = V(_a.safe_at(corner+cv::Vec3i(0,1,0)));
        V c110 = V(_a.safe_at(corner+cv::Vec3i(1,1,0)));
        V c001 = V(_a.safe_at(corner+cv::Vec3i(0,0,1)));
        V c101 = V(_a.safe_at(corner+cv::Vec3i(1,0,1)));
        V c011 = V(_a.safe_at(corner+cv::Vec3i(0,1,1)));
        V c111 = V(_a.safe_at(corner+cv::Vec3i(1,1,1)));

        V c00 = (V(1)-fc[2])*c000 + fc[2]*c001;
        V c01 = (V(1)-fc[2])*c010 + fc[2]*c011;
        V c10 = (V(1)-fc[2])*c100 + fc[2]*c101;
        V c11 = (V(1)-fc[2])*c110 + fc[2]*c111;

        V c0 = (V(1)-fc[1])*c00 + fc[1]*c01;
        V c1 = (V(1)-fc[1])*c10 + fc[1]*c11;

        *out = (V(1)-fc[0])*c0 + fc[0]*c1;

        // std::cout << fc[0] << " from " << c0 << " to " << c1 << f << corner << std::endl;
    }
    double  val(const double &v) const { return v; }
    template< typename JetT>
    double  val(const JetT &v) const { return v.a; }
    // static template <typename E> lin_int(E &loc, int max, )
    cv::Vec3i _shape;
};
