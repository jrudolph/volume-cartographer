#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/io/PointSetIO.hpp"

#include <unordered_map>
#include <filesystem>
#include <omp.h>

#include "vc/core/types/ChunkedTensor.hpp"

using shape = z5::types::ShapeType;
using namespace xt::placeholders;
namespace fs = std::filesystem;

using json = nlohmann::json;

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

std::string time_str()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y%m%d%H%M%S"); // HH:MM:SS
    oss << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

template <typename T, typename I>
float get_val(I &interp, cv::Vec3d l) {
    T v;
    interp.Evaluate(l[2], l[1], l[0], &v);
    return v;
}

int main(int argc, char *argv[])
{
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <output> <seg-path> <tgt-scale>" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path vol_path = argv[1];
    fs::path tgt_fn = argv[2];
    fs::path seg_path = argv[3];
    float tgt_scale = atof(argv[4]);

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "1", json::parse(std::ifstream(vol_path/"1/.zarray")).value<>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 1 " << ds->shape() << std::endl;
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
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    cv::resize(points, points, {0,0}, tgt_scale/surf->_scale[0], tgt_scale/surf->_scale[1], cv::INTER_CUBIC);

    cv::Mat_<uint8_t> img;

    points = points*0.5;

    readInterpolated3D(img, ds.get(), points, &chunk_cache);

    cv::imwrite(tgt_fn, img);


    return EXIT_SUCCESS;
}