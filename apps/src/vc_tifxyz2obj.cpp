#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

using json = nlohmann::json;

int get_add_vertex(std::ofstream &out, cv::Mat_<cv::Vec3f> &points, cv::Mat_<int> idxs, int &v_idx, cv::Vec2i loc)
{
    if (idxs(loc) == -1) {
        idxs(loc) = v_idx++;
        cv::Vec3f p = points(loc);
        out << "v " << p[0] << " " << p[1] << " " << p[2] << std::endl;
    }

    return idxs(loc);
}

void surf_write_obj(QuadSurface *surf, const fs::path &out_fn)
{
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    cv::Mat_<int> idxs(points.size(), -1);
    
    std::ofstream out(out_fn);
    
    int v_idx = 1;
    for(int j=0;j<points.rows-1;j++)
        for(int i=0;i<points.cols-1;i++)
            if (loc_valid(points, {j,i}))
            {
                int c00 = get_add_vertex(out, points, idxs, v_idx, {j,i});
                int c01 = get_add_vertex(out, points, idxs, v_idx, {j,i+1});
                int c10 = get_add_vertex(out, points, idxs, v_idx, {j+1,i});
                int c11 = get_add_vertex(out, points, idxs, v_idx, {j+1,i+1});
                
                out << "f " << c10 << " " << c00 << " " << c01 << " " << std::endl;
                out << "f " << c10 << " " << c01 << " " << c11 << " " << std::endl;
            }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <obj>" << std::endl;
        return EXIT_SUCCESS;
    }
    
    fs::path seg_path = argv[1];
    fs::path obj_path = argv[2];
    
    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    surf_write_obj(surf, obj_path);
    
    return EXIT_SUCCESS;
}