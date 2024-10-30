#pragma once

#include <opencv2/core.hpp>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

struct CeresGrid2DcvMat3f {
    enum { DATA_DIMENSION = 3 };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        cv::Vec3f v = _m(row, col);
        f[0] = v[0];
        f[1] = v[1];
        f[2] = v[2];
    }
    const cv::Mat_<cv::Vec3f> _m;
};

struct CeresGrid2DcvMat1f {
    enum { DATA_DIMENSION = 1 };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        cv::Vec3f v = _m(row, col);
        f[0] = v[0];
    }
    const cv::Mat_<float> _m;
};

template <typename B, int N>
struct CeresGrid2DcvMat_ {
    using V = cv::Vec<B,N>;
    enum { DATA_DIMENSION = N };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        const V &v = _m(row, col);
        for(int i=0;i<N;i++)
            f[i] = v[i];
    }
    const cv::Mat_<V> _m;
};


//cost functions for physical paper
struct DistLoss {
    DistLoss(float dist, float w) : _d(dist), _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        //FIXME where are invalid coords coming from?
        if (a[0] == -1 && a[1] == -1 && a[2] == -1) {
            residual[0] = T(0);
            return true;
        }
        //FIXME where are invalid coords coming from?
        if (b[0] == -1 && b[1] == -1 && b[2] == -1) {
            residual[0] = T(0);
            return true;
        }

        T d[3];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];
        d[2] = a[2] - b[2];

        d[0] = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);

        residual[0] = T(_w)*(d[0]/T(_d) - T(1));

        return true;
    }

    double _d;
    double _w;

    static ceres::CostFunction* Create(float d, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(d, w));
    }
};

//cost functions for physical paper
struct LocMinDistLoss {
    // LocMinDistLoss(const cv::Vec2f &scale, float mindist, float w) : _scale(scale), {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        //FIXME showhow we sometimes feed invalid loc (-1/-1) probably because we encountered the edge?
        // if (a[0] < 0 || a[1] < 0 || b[0] < 0 || b[1] < 0) {
        //     residual[0] = T(0);
        //     return true;
        // }

        T as[2];
        T bs[2];

        T d[2];
        d[0] = (a[0]-b[0])/T(_scale[0]);
        d[1] = (a[1]-b[1])/T(_scale[1]);

        T d2 = d[0]*d[0] + d[1]*d[1];

        if (d2 < T(_mindist*_mindist))
            residual[0] = T(_mindist) - sqrt(d2);
        else
            residual[0] = T(0);

        return true;
    }

    cv::Vec2f _scale;
    float _mindist;
    float _w;

    static ceres::CostFunction* Create(const cv::Vec2f &scale, float mindist, float w)
    {
        return new ceres::AutoDiffCostFunction<LocMinDistLoss, 1, 2, 2>(new LocMinDistLoss({scale, mindist, w}));
    }
};

//cost functions for physical paper
struct StraightLoss {
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T d1[3], d2[3];
        d1[0] = b[0] - a[0];
        d1[1] = b[1] - a[1];
        d1[2] = b[2] - a[2];

        d2[0] = c[0] - b[0];
        d2[1] = c[1] - b[1];
        d2[2] = c[2] - b[2];

        T l1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]);
        T l2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]);

        T dot = (d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2])/(l1*l2);

        residual[0] = T(0.3)*(T(1)-dot);

        return true;
    }

    static ceres::CostFunction* Create()
    {
        return new ceres::AutoDiffCostFunction<StraightLoss, 1, 3, 3, 3>(new StraightLoss());
    }
};

//cost functions for physical paper
struct SurfaceLoss {
    SurfaceLoss(const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, const T* const l, T* residual) const {
        T v[3];

        _interpolator.Evaluate(l[1], l[0], v);

        residual[0] = T(_w)*(v[0] - p[0]);
        residual[1] = T(_w)*(v[1] - p[1]);
        residual[2] = T(_w)*(v[2] - p[2]);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, float w = 1.0)
    {
        // auto l = new SurfaceLoss(grid);
        // std::cout << l->_interpolator.grid_._m.size() << std::endl;
        // auto g = CeresGrid2DcvMat3f({grid});
        // std::cout << g._m.size() << std::endl;
        // std::cout << l->_interpolator.grid_._m.size() << std::endl;
        return new ceres::AutoDiffCostFunction<SurfaceLoss, 3, 3, 2>(new SurfaceLoss(interp, w));
    }

    const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &_interpolator;
};

//cost functions for physical paper
struct UsedSurfaceLoss {
    UsedSurfaceLoss(const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v[1];

        _interpolator.Evaluate(l[1], l[0], v);

        residual[0] = T(_w)*(v[0]);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &interp, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<UsedSurfaceLoss, 3, 2>(new UsedSurfaceLoss(interp, w));
    }

    const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &_interpolator;
};


struct VXYCost {
    VXYCost(const std::pair<cv::Vec2i, cv::Vec3f> &p1, const std::pair<cv::Vec2i, cv::Vec3f> &p2) : _p1(p1.second), _p2(p2.second)
    {
        _d = p2.first-p1.first;
    };
    template <typename T>
    bool operator()(const T* const vx, const T* const vy, T* residual) const {
        T p1[3] = {T(_p1[0]),T(_p1[1]),T(_p1[2])};
        T p2[3];

        p2[0] = p1[0] + T(_d[0])*vx[0] + T(_d[1])*vy[0];
        p2[1] = p1[1] + T(_d[0])*vx[1] + T(_d[1])*vy[1];
        p2[2] = p1[2] + T(_d[0])*vx[2] + T(_d[1])*vy[2];

        residual[0] = p2[0] - T(_p2[0]);
        residual[1] = p2[1] - T(_p2[1]);
        residual[2] = p2[2] - T(_p2[2]);

        return true;
    }

    cv::Vec3f _p1, _p2;
    cv::Vec2i _d;

    static ceres::CostFunction* Create(const std::pair<cv::Vec2i, cv::Vec3f> &p1, const std::pair<cv::Vec2i, cv::Vec3f> &p2)
    {
        return new ceres::AutoDiffCostFunction<VXYCost, 3, 3, 3>(new VXYCost(p1, p2));
    }
};

struct OrthogonalLoss {
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        T dot;
        dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

        T la = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        T lb = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);

        residual[0] = dot/(la*lb);

        return true;
    }

    static ceres::CostFunction* Create()
    {
        return new ceres::AutoDiffCostFunction<OrthogonalLoss, 1, 3, 3>(new OrthogonalLoss());
    }
};

struct ParallelLoss {
    ParallelLoss(const cv::Vec3f &ref, float w) : _w(w)
    {
        cv::normalize(ref, _ref);
    }

    template <typename T>
    bool operator()(const T* const a, T* residual) const {
        T dot;
        dot = a[0]*T(_ref[0]) + a[1]*T(_ref[1]) + a[2]*T(_ref[2]);

        T la = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);

        residual[0] = T(1)-T(_w)*dot/la;

        return true;
    }

    cv::Vec3f _ref;
    float _w;

    static ceres::CostFunction* Create(const cv::Vec3f &ref, const float &w)
    {
        return new ceres::AutoDiffCostFunction<ParallelLoss, 1, 3>(new ParallelLoss(ref, w));
    }
};

//can't decide between C++ tensor libs, xtensor wasn't great and few algos supported anyways...
template <typename T>
class StupidTensor
{
public:
    StupidTensor() {};
    template <typename O> StupidTensor(const StupidTensor<O> &other) { create(other.planes[0].size(), other.planes.size()); };
    StupidTensor(const cv::Size &size, int d) { create(size, d); };
    void create(const cv::Size &size, int d)
    {
        planes.resize(d);
        for(auto &p : planes)
            p.create(size);
    }
    void setTo(const T &v)
    {
        for(auto &p : planes)
            p.setTo(v);
    }
    template <typename O> void convertTo(O &out, int code) const
    {
        out.create(planes[0].size(), planes.size());
        for(int z=0;z<planes.size();z++)
            planes[z].convertTo(out.planes[z], code);
    }
    T &at(int z, int y, int x) { return planes[z](y,x); }
    T &operator()(int z, int y, int x) { return at(z,y,x); }
    std::vector<cv::Mat_<T>> planes;
};

using st_u = StupidTensor<uint8_t>;
using st_f = StupidTensor<cv::Vec<float,1>>;
using st_1u = StupidTensor<cv::Vec<uint8_t,1>>;
using st_3f = StupidTensor<cv::Vec3f>;

template <typename B, int N>
class StupidTensorInterpolator
{
using V = cv::Vec<B,N>;
public:
    using GRID = CeresGrid2DcvMat_<B,N>;
    StupidTensorInterpolator() {};
    StupidTensorInterpolator(StupidTensor<V> &t)
    {
        _d = t.planes.size();
        for(auto &p : t.planes)
            grid_planes.push_back(GRID({p}));
        for(auto &p : grid_planes)
            interp_planes.push_back(ceres::BiCubicInterpolator<GRID>(p));
    };
    std::vector<GRID> grid_planes;
    std::vector<ceres::BiCubicInterpolator<GRID>> interp_planes;
    template <typename T> void Evaluate(const T &z, const T &y, const T &x, T *out) const
    {
        //FIXME linear interpolate along z
        if (z < 0.0)
            interp_planes[0].Evaluate(y, x, out);
        else if (z >= _d-2)
            return interp_planes[_d-1].Evaluate(y, x, out);
        else {
            T m = z-floor(z);
            int zi = idx(z);
            T low[N];
            T high[N];
            interp_planes[zi].Evaluate(y, x, low);
            interp_planes[zi+1].Evaluate(y, x, high);
            for(int i=0;i<N;i++)
                out[i] = (T(1)-m)*low[i] + m*high[i];
        }
    }
    int  idx(const double &v) const { return v; }
    template< typename JetT>
    int  idx(const JetT &v) const { return v.a; }
    int _d = 0;
};

//cost functions for physical paper
struct EmptySpaceLoss {
    EmptySpaceLoss(const StupidTensorInterpolator<uint8_t,1> &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v;

        _interpolator.Evaluate<T>(l[2], l[1], l[0], &v);

        residual[0] = T(_w)/(v+T(0.3));

        return true;
    }

    float _w;
    const StupidTensorInterpolator<uint8_t,1> &_interpolator;

    static ceres::CostFunction* Create(const StupidTensorInterpolator<uint8_t,1> &interp, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLoss, 1, 3>(new EmptySpaceLoss(interp, w));
    }

};

//cost functions for physical paper
struct EmptySpaceLineLoss {
    // EmptySpaceMultiLoss(const StupidTensorInterpolator<uint8_t,1> &interp, int steps, uint8_t *state_a, uint8_t *state_b,  w) : _interpolator(interp), _steps(steps), _w(w) {};
    template <typename T>
    bool operator()(const T* const la, const T* const lb, T* residual) const {
        T v;
        T sum = T(0);

        for(int i=1;i<_steps;i++) {
            T f2 = T(float(i)/_steps);
            T f1 = T(1.0f-float(i)/_steps);
            _interpolator.Evaluate<T>(f1*la[2]+f2*lb[2], f1*la[1]+f2*lb[1], f1*la[0]+f2*lb[0], &v);
            sum += T(_w)/(v+T(0.3));
        }

        residual[0] = sum/T(_steps-1);

        return true;
    }

    const StupidTensorInterpolator<uint8_t,1> &_interpolator;
    int _steps;
    float _w;

    static ceres::CostFunction* Create(const StupidTensorInterpolator<uint8_t,1> &interp, int steps, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLineLoss, 1, 3, 3>(new EmptySpaceLineLoss({interp, steps, w}));
    }

};
