#include "OpChain.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

// QuadSurface *OpChain::surf(SurfacePointer *ptr, const cv::Size &size, float scale, ChunkCache *cache, z5::Dataset *ds)
// {
//     QuadSurface *last = nullptr;
//
//     if (_src_mode == OpChainSourceMode::RAW) {
//         last = _src;
//     }
//     else if (_src_mode == OpChainSourceMode::DEFAULT_MESHING)
//     {
//         //re-parametrize base then go trough surfaces ...
//         if (_crop) {
//             delete _crop;
//             _crop = nullptr;
//         }
//
//         int search_step = 40;
//         int mesh_step = 20;
//
//         _crop = regularized_local_quad(_src, ptr, size.width/mesh_step/scale, size.height/mesh_step/scale, search_step, mesh_step);
//     }
//
//     //reset op chain
//     for(auto s : _ops) {
//         s->setBase(last);
//         last = s;
//     }
//
//     return last;
// }

void OpChain::append(DeltaQuadSurface *op)
{
    _ops.push_back(op);
}

SurfacePointer *OpChain::pointer()
{
    return _src->pointer();
}

void OpChain::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    _src->move(ptr, offset);
}

bool OpChain::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _src->valid(ptr, offset);
}

cv::Vec3f OpChain::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _src->loc(ptr, offset);
}

cv::Vec3f OpChain::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //FIXME use cached surface? Or regen
    return _src->coord(ptr, offset);
}

cv::Vec3f OpChain::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //FIXME use cached surface? Or regen
    return _src->normal(ptr, offset);
}

float OpChain::pointTo(SurfacePointer *ptr, const cv::Vec3f &coord, float th)
{
    //FIXME use cached surf? Or use src surface?
    return _src->pointTo(ptr, coord, th);
}

void OpChain::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    QuadSurface *last = nullptr;
    SurfacePointer *ptr_center = ptr;
    if (!ptr_center)
        ptr_center = _src->pointer();

    if (_crop) {
        delete _crop;
        _crop = nullptr;
    }

    if (_src_mode == OpChainSourceMode::RAW) {
        last = _src;
        ptr_center = ptr_center->clone();
        _src->move(ptr_center, offset+cv::Vec3f(-size.width/2,-size.height/2,0));
    }
    else if (_src_mode == OpChainSourceMode::DEFAULT_MESHING)
    {
        int search_step = 100;
        int mesh_step = 5;

        _crop = regularized_local_quad(_src, ptr, size.width/mesh_step/scale, size.height/mesh_step/scale, search_step, mesh_step);
        last = _crop;
    }

    //reset op chain
    for(auto s : _ops) {
        s->setBase(last);
        last = s;
    }

    if (_src_mode == OpChainSourceMode::RAW) {
        last->gen(coords, normals, size, ptr, scale, offset);
    }
    else
        last->gen(coords, normals, size, nullptr, scale, {-size.width/2, -size.height/2, ((TrivialSurfacePointer*)ptr_center)->loc[2]});
}

const char *op_name(Surface *op)
{
    if (!op)
        return "";

    if (dynamic_cast<OpChain*>(op))
        return "source";
    if (dynamic_cast<RefineCompSurface*>(op))
        return "refineAlphaComp";
    return "FIXME unknown op name";
}
