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

int main(int argc, char *argv[])
{
    if (argc != 6) {
        std::cout << "usage: " << argv[0] << " <zarr-volume> <json-params> <seed-x> <seed-y> <seed-z>" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path vol_path = argv[1];
    const char *params_path = argv[2];

    cv::Vec3f origin = {atof(argv[3]),atof(argv[4]),atof(argv[5])};

    std::ifstream params_f(params_path);
    json params = json::parse(params_f);

    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "0", nlohmann::json::parse(std::ifstream(vol_path/"0/.zarray")).value<>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    ChunkCache chunk_cache(2e9);

    QuadSurface *surf = empty_space_tracing_quad_phys(ds.get(), 1.0, &chunk_cache, origin, 20);
}
