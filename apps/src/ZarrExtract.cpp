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

#include "vc/core/util/Slicing.hpp"

#include <unordered_map>

using shape = z5::types::ShapeType;
using namespace xt::placeholders;

std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const std::vector<int> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

template <size_t N>
std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}


shape chunkId(const std::unique_ptr<z5::Dataset> &ds, shape coord)
{
    shape div = ds->chunking().blockShape();
    shape id = coord;
    for(int i=0;i<id.size();i++)
        id[i] /= div[i];
    return id;
}

shape idCoord(const std::unique_ptr<z5::Dataset> &ds, shape id)
{
    shape mul = ds->chunking().blockShape();
    shape coord = id;
    for(int i=0;i<coord.size();i++)
        coord[i] *= mul[i];
    return coord;
}

class CacheEntry
{
public:
    std::vector<std::uint8_t> data;
    z5::types::ShapeType shape;
    std::size_t size;
};

size_t cache_hits = 0;
size_t cache_miss = 0;

//lazy for now, for 3d chunk cache this should be good enough.
std::unordered_map<uint64_t,CacheEntry> cache;


void putCache(z5::types::ShapeType chunkId, void* chunk, z5::types::ShapeType chunkShape, std::size_t chunkSize)
{
    uint64_t key = (chunkId[0]) ^ (chunkId[1]<<24) ^ (chunkId[2]<<48);
    if (cache.find(key) != cache.end())
        return;
    
    CacheEntry entry = {*static_cast<std::vector<std::uint8_t>*>(chunk), chunkShape, chunkSize};
    cache[key] = entry;
}

void *pullCache(z5::types::ShapeType chunkId, z5::types::ShapeType& chunkShape, std::size_t& chunkSize)
{
    uint64_t key = (chunkId[0]) ^ (chunkId[1]<<24) ^ (chunkId[2]<<48);
    if (cache.find(key) == cache.end()) {
        cache_miss++;
        return nullptr;
    }
    
    cache_hits++;
    CacheEntry &entry = cache[key];
    chunkShape = entry.shape;
    chunkSize = entry.size;
    return &entry.data;
}

int main(int argc, char *argv[])
{
  assert(argc == 2);
  // z5::filesystem::handle::File f(argv[1]);
  z5::filesystem::handle::Group group(argv[1], z5::FileMode::FileMode::r);
  z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
  std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

  std::cout << "ds shape " << ds->shape() << std::endl;
  std::cout << "ds shape via chunk " << ds->chunking().shape() << std::endl;
  std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

  ds->enableCaching(true, putCache, pullCache);
  
  
  
  //read the chunk around pixel coord
  // shape coord = shape({0,2000,2000});

  // shape id = chunkId(ds, coord);

  // fs::path path;
  // ds->chunkPath(id, path);

//   std::cout << "id " << id << " path " << path << std::endl;
// 
//   int threads = static_cast<int>(std::thread::hardware_concurrency());
// 
//   z5::types::ShapeType offset = idCoord(ds, id);
//   xt::xarray<uint8_t>::shape_type shape = ds->chunking().blockShape();
//   shape[0] = 1;
//   xt::xarray<uint8_t> array(shape);
// 
//   z5::multiarray::readSubarray<uint8_t>(*ds, array, offset.begin(), threads);
// 
//   cv::Mat m = cv::Mat(shape[1], shape[2], CV_8U, array.data());
// 
//   cv::imwrite("extract.tif", m);
// 
  xt::xarray<float> coords = xt::zeros<float>({500,500,3});
  xt::xarray<uint8_t> img;
//   
//   cv::Vec3d slice_center = {0.5,0.5,0.5};
//   cv::Vec3d slice_normal = {0.5,0.5,0.5};
  
//   for(int y=0;y<500;y++)
//       for(int x=0;x<500;x++) {
//           coords(y,x,0) = 500;
//           coords(y,x,1) = y+2000;
//           coords(y,x,2) = x+2000;
//       }
// 
//   readInterpolated3D(img,ds,coords);
//   m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
//   cv::imwrite("img.tif", m);
// 
// 
//   readInterpolated3DChunked(img,ds,coords,256);
//   
//   
//   m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
//   
//   cv::imwrite("img2.tif", m);
  
  PlaneCoords gen_plane({2000,1800,2000},{0.0,0.1,1.0});
  
  gen_plane.gen_coords(coords, 4000, 4000);
  
  readInterpolated3DChunked(img,ds.get(),coords,256);
  cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  cv::imwrite("plane.tif", m);
  
  printf("cache hit/miss %d %d %.3f",cache_hits,cache_miss,float(cache_hits)/(cache_hits+cache_miss));
  
  // m = cv::Mat(coords.shape(0), coords.shape(1), CV_32FC3, coords.data());
  // std::vector<cv::Mat> chs;
  // cv::split(m, chs);
  // cv::imwrite("coords_x.tif", chs[2]);
  // cv::imwrite("coords_y.tif", chs[1]);
  // cv::imwrite("coords_z.tif", chs[0]);
    
  return 0;
}