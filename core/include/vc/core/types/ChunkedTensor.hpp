#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>

#include "z5/multiarray/xtensor_access.hxx"

#include <random>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

// static std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }
//
// template <size_t N>
// static std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }

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

static uint64_t miss = 0;
static uint64_t total = 0;
static uint64_t chunk_compute_collisions = 0;
static uint64_t chunk_compute_total = 0;

template <typename T, typename C> class Chunked3dAccessor;

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;
    // friend Chunked3dAccessor;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        auto shape = ds->chunking().blockShape();
        _border = compute_f.BORDER;
    };
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);
        return _chunks[id]->operator()(p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s);
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        CHUNKT *chunk = nullptr;

        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk->operator()(p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s);
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    CHUNKT *cache_chunk_safe_ram(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);
        CHUNKT *small = new CHUNKT();
        *small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            _compute_f.template compute<CHUNKT,T>(large, *small);

            _mutex.lock();
            if (!_chunks.count(id))
                _chunks[id] = small;
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                delete small;
                small = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

        return small;
    }

    fs::path id_path(const fs::path &dir, const cv::Vec3i &id)
    {
        return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
    }

    CHUNKT *cache_chunk_safe_mmap(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT *small = new CHUNKT();

        fs::path tgt_path = id_path(_compute_f.CHUNK_DIR, id);
        size_t len = s*s*s;

        if (fs::exists(tgt_path)) {
            int fd = open(tgt_path.string().c_str(), O_RDWR);
            uint8_t *mem = (uint8_t*)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            *small = xt::empty<T>({s,s,s});
            for(int i=0;i<len;i++)
                (&(*small)(0,0,0))[i] = mem[i];

            _mutex.lock();
            if (!_chunks.count(id))
                _chunks[id] = small;
            else {
#pragma omp atomic
                chunk_compute_collisions++;
                munmap(mem, len);
                delete small;
                small = _chunks[id];
            }
#pragma omp atomic
            chunk_compute_total++;
            _mutex.unlock();

            return small;
        }

        fs::path tmp_path;
        _mutex.lock();
        tmp_path = fs::path(_compute_f.CHUNK_DIR) / std::to_string(_tmp_counter++);
        _mutex.unlock();
        int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        ftruncate(fd, len);
        uint8_t *mem = (uint8_t*)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);


        *small = xt::empty<T>({s,s,s});

        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

        readArea3D(large, offset, _ds, _cache);

        _compute_f.template compute<CHUNKT,T>(large, *small);

        for(int i=0;i<len;i++)
            mem[i] = (&(*small)(0,0,0))[i];

        _mutex.lock();

        if (!_chunks.count(id)) {
            _chunks[id] = small;
            int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

            if (ret)
                throw std::runtime_error("oops rename failed!");
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            munmap(mem, len);
            delete small;
            unlink(tmp_path.string().c_str());
            small = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return small;
    }

    CHUNKT *cache_chunk_safe_alloc(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);
        CHUNKT *small = new CHUNKT();
        *small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            // void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

            _compute_f.template compute<CHUNKT,T>(large, *small);

            _mutex.lock();
            if (!_chunks.count(id))
                _chunks[id] = small;
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            delete small;
            small = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return small;
    }

    CHUNKT *cache_chunk_safe(const cv::Vec3i &id)
    {
        if (_compute_f.CHUNK_DIR.size())
            return cache_chunk_safe_mmap(id);
        else
            return cache_chunk_safe_alloc(id);
    }

    CHUNKT *cache_chunk(const cv::Vec3i &id) {
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

    CHUNKT *chunk(const cv::Vec3i &id) {
        if (!_chunks.count(id))
            return cache_chunk(id);
        return _chunks[id];
    }

    CHUNKT *chunk_safe(const cv::Vec3i &id) {
        CHUNKT *chunk = nullptr;
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

    std::unordered_map<cv::Vec3i,CHUNKT*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
    uint64_t _tmp_counter = 0;
};

void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << double(miss)/total << std::endl;
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << double(chunk_compute_collisions)/chunk_compute_total << std::endl;
}

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

        return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
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

        return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
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
    CHUNKT *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

