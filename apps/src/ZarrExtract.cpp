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

//NOTE depending on request this might load a lot (the whole array) into RAM
void readInterpolated3D(xt::xarray<uint8_t> &out, std::unique_ptr<z5::Dataset> &ds, const xt::xarray<float> &coords)
{
    // auto dims = xt::range(_,coords.shape().size()-2);
    // std::vector<long int> da = dims;
    std::vector<int> dims;
    for(int i=0;i<coords.shape().size()-1;i++)
        dims.push_back(i);
    xt::xarray<float> upper = xt::amax(coords, dims);
    xt::xarray<float> lower = xt::amin(coords, dims);
    
    std::cout << "lwo/high" << lower << upper << std::endl;
    upper(0) = std::min(upper(0),float(ds->shape(0)));
    upper(1) = std::min(upper(1),float(ds->shape(1)));
    upper(2) = std::min(upper(2),float(ds->shape(2)));
    lower(0) = std::max(lower(0),0.0f);
    lower(1) = std::max(lower(1),0.0f);
    lower(2) = std::max(lower(2),0.0f);
    // lower = xt::amax(lower, {0,0,0});
    // std::cout << "lwo/high" << lower << upper << std::endl;
    
    // std::cout << "maxshape" << .shape() << std::endl;
    
    shape offset(3);
    shape size(3);
    for(int i=0;i<3;i++) {
        offset[i] = lower[i];
        size[i] = ceil(std::max(upper[i] - offset[i]+1,1.0f));
    }
    std::cout << "offset" << offset << std::endl;
    std::cout << "size" << size << std::endl;
    
    if (!size[0] || !size[1] || !size[2])
        return;
    
    xt::xarray<uint8_t> buf(size);
    
    z5::multiarray::readSubarray<uint8_t>(*ds, buf, offset.begin(), std::thread::hardware_concurrency());
    
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    if (out.shape() != out_shape) {
        std::cout << "allocating out as its wrong size!" << std::endl;
        out = xt::zeros<uint8_t>(out_shape);
    }
    
    auto iter_coords = xt::axis_slice_begin(coords, 2);
    auto iter_out = xt::axis_slice_begin(out, 2);
    auto end_coords = xt::axis_slice_end(coords, 2);
    auto end_out = xt::axis_slice_end(out, 2);
    size_t inb_count = 0;
    size_t total = 0;
    while(iter_coords != end_coords)
    {
        total++;
        std::vector<int> idx = {(*iter_coords)(0)-offset[0],(*iter_coords)(1)-offset[1],(*iter_coords)(2)-offset[2]};
        // if (total % 1000 == 0)
            // std::cout << idx << *iter_coords << offset << buf.in_bounds(idx[0],idx[1],idx[2]) << buf.shape() << "\n";
        if (buf.in_bounds(idx[0],idx[1],idx[2])) {
            *iter_out = buf[idx];
            inb_count++;
        }
        else
            *iter_out = 0;
        iter_coords++;
        iter_out++;
    }
    // printf("inb sum: %d %d %.f\n", inb_count, total, float(inb_count)/total);
    
    // std::cout << "read to buf" << std::endl;
}

//TODO make the chunking more intelligent and efficient - for now this is probably good enough ...
//this method will chunk over the second and third last dim of coords (which should probably be x and y)
void readInterpolated3DChunked(xt::xarray<uint8_t> &out, std::unique_ptr<z5::Dataset> &ds, const xt::xarray<float> &coords, size_t chunk_size)
{
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    out = xt::zeros<uint8_t>(out_shape);
    
    std::cout << out_shape << " " << coords.shape() << "\n";
    
    //FIXME assert dims
    
    int xdim = coords.shape().size()-2;
    int ydim = coords.shape().size()-3;
    
    std::cout << coords.shape(ydim) << " " << coords.shape(xdim) << "\n";

    for(size_t y = 0;y<coords.shape(ydim);y+=chunk_size)
        for(size_t x = 0;x<coords.shape(xdim);x+=chunk_size) {
            xt::xarray<uint8_t> out_view = xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()});
            const xt::xarray<float> coord_view = xt::strided_view(coords, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()});
            
            std::cout << out_view.shape() << " " << x << "x" << y << std::endl;
            // return;
            xt::xarray<uint8_t> tmp;
            readInterpolated3D(tmp, ds, coord_view);
            //FIXME figure out xtensor copy/reference dynamics ...
            xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()}) = tmp;
        }
        
}

class CoordGenerator
{
public:
    //given input volume shape, fill a coord slice
    virtual void gen_coords(xt::xarray<float> &coords, int w, int h) = 0;
};

class PlaneCoords : public CoordGenerator
{
public:
    PlaneCoords(cv::Vec3f origin_, cv::Vec3f normal_) : origin(origin_)
    {
        cv::normalize(normal_, normal, 1,0, cv::NORM_L2);
                                 
    };
    virtual void gen_coords(xt::xarray<float> &coords, int w, int h)
    {
        // auto grid = xt::meshgrid(xt::arange<float>(0,h),xt::arange<float>(0,w));

        cv::Vec3f vx,vy;
        //TODO will there be a jump around the midpoint?
        //FIXME how to decide direction of cross vector?
        if (abs(normal[0]) >= abs(normal[1])) {
            vx = cv::Vec3f(1,0,origin[2]-normal[0]/normal[2]);
            cv::normalize(vx, vx, 1,0, cv::NORM_L2);
            vy = cv::Mat(normal).cross(cv::Mat(vx));
        }
        else {
            vy = cv::Vec3f(0,1,origin[2]-normal[1]/normal[2]);
            cv::normalize(vy, vy, 1,0, cv::NORM_L2);
            vx = cv::Mat(normal).cross(cv::Mat(vy));
        }
        if (vx[0] < 0)
            vx *= -1;
        if (vy[10] < 0)
            vy *= -1;
        
        std::cout << "vecs" << normal << vx << vy << "\n";
        
        xt::xarray<float> vx_t{{{vx[2],vx[1],vx[0]}}};
        xt::xarray<float> vy_t{{{vy[2],vy[1],vy[0]}}};
        xt::xarray<float> origin_t{{{origin[2],origin[1],origin[0]}}};
        
        xt::xarray<float> xrange = xt::arange<float>(-w/2,w/2).reshape({1, -1, 1});
        xt::xarray<float> yrange = xt::arange<float>(-h/2,h/2).reshape({-1, 1, 1});
        
        // xrange = xrange.reshape(-1, 1, 1);
        
        std::cout << xrange.shape() << vx_t.shape() <<  std::endl;
        
        coords = vx_t*xrange + vy_t*yrange+origin_t;
        
        
        // auto res = xt::stack(xt::xtuple(std::get<0>(grid),std::get<1>(grid)), 2);
        // coords = res;
        // // auto sh = ;
        std::cout << coords.shape() << std::endl;
        std::cout << xt::strided_view(coords,{h/2,w/2,xt::all()}) << std::endl;
        std::cout << xt::strided_view(coords,{h/2,w/2+1,xt::all()}) << std::endl;
        std::cout << xt::strided_view(coords,{h/2+1,w/2,xt::all()}) << std::endl;
        
        
        std::cout << xt::strided_view(coords,{0,0,xt::all()}) << std::endl;
        std::cout << xt::strided_view(coords,{0,w-1,xt::all()}) << std::endl;
        std::cout << xt::strided_view(coords,{h-1,0,xt::all()}) << std::endl;
        std::cout << xt::strided_view(coords,{h-1,w-1,xt::all()}) << std::endl;
    }
private:
    cv::Vec3f origin = {0,0,0};
    cv::Vec3f normal = {1,1,1};
};

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

  //read the chunk around pixel coord
  shape coord = shape({0,2000,2000});

  shape id = chunkId(ds, coord);

  fs::path path;
  ds->chunkPath(id, path);

  std::cout << "id " << id << " path " << path << std::endl;

  int threads = static_cast<int>(std::thread::hardware_concurrency());

  z5::types::ShapeType offset = idCoord(ds, id);
  xt::xarray<uint8_t>::shape_type shape = ds->chunking().blockShape();
  shape[0] = 1;
  xt::xarray<uint8_t> array(shape);

  z5::multiarray::readSubarray<uint8_t>(*ds, array, offset.begin(), threads);

  cv::Mat m = cv::Mat(shape[1], shape[2], CV_8U, array.data());

  cv::imwrite("extract.tif", m);

  xt::xarray<float> coords = xt::zeros<float>({500,500,3});
  xt::xarray<uint8_t> img;
  
  cv::Vec3d slice_center = {0.5,0.5,0.5};
  cv::Vec3d slice_normal = {0.5,0.5,0.5};
  
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
  
  PlaneCoords gen_plane({2000,1800,0},{0.0,0.1,1.0});
  
  
  gen_plane.gen_coords(coords, 1024, 768);
  
  readInterpolated3DChunked(img,ds,coords,256);
  // readInterpolated3D(img,ds,coords);
  m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  cv::imwrite("plane.tif", m);
  
  m = cv::Mat(coords.shape(0), coords.shape(1), CV_32FC3, coords.data());
  std::vector<cv::Mat> chs;
  cv::split(m, chs);
  cv::imwrite("coords_x.tif", chs[2]);
  cv::imwrite("coords_y.tif", chs[1]);
  cv::imwrite("coords_z.tif", chs[0]);
    
  return 0;
}
