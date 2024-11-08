#pragma once

#include "vc/core/util/Surface.hpp"

#include <set>

class QuadSurface;
class DeltaSurface;
class SurfacePointer;
class ChunkCache;
class FormSetSrc;

namespace z5 {
    class Dataset;
}

enum class OpChainSourceMode: int
{
    RAW = 0,
    BLUR = 1,
    GREEDY = 2
};

//special "windowed" surface that represents a set of delta surfaces on top of a base QuadSurface
//caches the generated coords to base surface method on this cached representation
class OpChain : public Surface {
public:
    OpChain(QuadSurface *src) : _src(src) { if (src->rawPoints().rows < 1000) _src_mode = OpChainSourceMode::RAW; };
    // cv::Mat render(SurfacePointer *ptr, const cv::Size &size, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    QuadSurface *surf(SurfacePointer *ptr, const cv::Size &size, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    void append(DeltaSurface *op);

    SurfacePointer *pointer() override;
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &coord, float th, int max_iters = 1000)  override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset);
    bool slow() { return _src_mode == OpChainSourceMode::GREEDY || (_src_mode == OpChainSourceMode::BLUR && !_src_blur); }

    std::vector<DeltaSurface*> ops() { return _ops; };

    void setEnabled(DeltaSurface *surf, bool enabled);
    bool enabled(DeltaSurface *surf);
    QuadSurface *src() { return _src; }


    friend class FormSetSrc;

protected:
    OpChainSourceMode _src_mode = OpChainSourceMode::BLUR;
    std::vector<DeltaSurface*> _ops;
    std::set<DeltaSurface*> _disabled;
    QuadSurface *_src = nullptr;
    QuadSurface *_crop = nullptr;
    QuadSurface *_src_blur = nullptr;
};

const char * op_name(Surface *op);
