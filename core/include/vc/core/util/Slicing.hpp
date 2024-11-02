#pragma once

#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>

#include <shared_mutex>

namespace z5
{
    class Dataset;
}

struct vec4i_hash {
    size_t operator()(cv::Vec4i p) const
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);
        size_t hash4 = std::hash<int>{}(p[3]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

//TODO generation overrun
//TODO groupkey overrun
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {};
    ~ChunkCache();
    
    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    int groupIdx(std::string name);
    
    //key should be unique for chunk and contain groupkey (groupkey sets highest 16bits of uint64_t)
    void put(cv::Vec4i key, xt::xarray<uint8_t> *ar);
    std::shared_ptr<xt::xarray<uint8_t>> get(cv::Vec4i key);
    void reset();
    bool has(cv::Vec4i idx);
    std::shared_mutex mutex;
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> _store;
    //store generation number
    std::unordered_map<cv::Vec4i,uint64_t,vec4i_hash> _gen_store;
    //store group keys
    std::unordered_map<std::string,int> _group_store;

    std::shared_mutex _mutex;
};

//NOTE depending on request this might load a lot (the whole array) into RAM
// void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr);
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache *cache);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);
