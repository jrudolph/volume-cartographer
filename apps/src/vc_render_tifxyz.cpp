#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

using json = nlohmann::json;

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]";
    }
    return out;
}

int main(int argc, char *argv[])
{
    if (argc != 6 && argc != 7) {
        std::cout << "usage: " << argv[0] << " <ome-arr-volume> <output> <seg-path> <tgt-scale> <ome-zarr-group-idx>" << std::endl;
        std::cout << "or: " << argv[0] << " <ome-zarr-volume> <ptn> <seg-path> <tgt-scale> <ome-zarr-group-idx> <num-slices>" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path vol_path = argv[1];
    const char *tgt_ptn = argv[2];
    fs::path seg_path = argv[3];
    float tgt_scale = atof(argv[4]);
    int group_idx = atoi(argv[5]);
    
    int num_slices = 1;
    if (argc == 7)
        num_slices = atoi(argv[6]);

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, std::to_string(group_idx), json::parse(std::ifstream(vol_path/std::to_string(group_idx)/".zarray")).value<>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group " << group_idx << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    ChunkCache chunk_cache(10e9);

    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }
    
    cv::Size tgt_size = surf->rawPoints().size();
    tgt_size.width *= tgt_scale/surf->_scale[0];
    tgt_size.height *= tgt_scale/surf->_scale[1];
    
    std::cout << "rendering size " << tgt_size << " at scale " << tgt_scale << std::endl;
    
    cv::Mat_<cv::Vec3f> points, normals;
    
    bool slice_gen = false;
    
    if (tgt_size.width >= 10000)
        slice_gen = true;
    else
        surf->gen(&points, &normals, tgt_size, nullptr, tgt_scale, {-tgt_size.width/2,-tgt_size.height/2,0});

    cv::Mat_<uint8_t> img;

    float ds_scale = pow(2,-group_idx);
    if (group_idx && !slice_gen) {
        points *= ds_scale;
    }

    if (num_slices == 1) {
        readInterpolated3D(img, ds.get(), points, &chunk_cache);
        cv::imwrite(tgt_ptn, img);
    }
    else {
        char buf[1024];
        for(int i=0;i<num_slices;i++) {
            float off = i-num_slices/2;
            if (slice_gen) {
                img.create(tgt_size);
                for(int x=0;x<tgt_size.width;x+=1024) {
                    int w = std::min(tgt_size.width-x, 1024);
                    surf->gen(&points, &normals, {w,tgt_size.height}, nullptr, tgt_scale, {-tgt_size.width/2+x,-tgt_size.height/2,0});
                    cv::Mat_<uint8_t> slice;
                    readInterpolated3D(slice, ds.get(), points*ds_scale+off*normals*ds_scale, &chunk_cache);
                    slice.copyTo(img(cv::Rect(x,0,w,tgt_size.height)));
                }
            }
                
            else {
                readInterpolated3D(img, ds.get(), points+off*ds_scale*normals, &chunk_cache);
            }
            snprintf(buf, 1024, tgt_ptn, i);
            cv::imwrite(buf, img);
        }
    }


    return EXIT_SUCCESS;
}
