///// Per-Pixel Map /////
//
// Effectively a raster of a UV Map for reverse lookups.
// Every pixel in the map holds the 3D position and normal
// used to generate that pixel's intensity in the corresponding
// texture image. Useful for regenerating textures with differing
// parameters or for identifying where in a volume a particular
// came from.
//
// Created by Seth Parker on 3/17/16.


#ifndef VC_PERPIXELMAP_H
#define VC_PERPIXELMAP_H

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace volcart {
    class PerPixelMap {
    public:
        ///// Constructors /////
        // Create empty
        PerPixelMap() : _width(0), _height(0){};

        // Create new
        PerPixelMap( int height, int width );

        // Construct map from file
        PerPixelMap( boost::filesystem::path path );

        ///// Check if initialized /////
        bool initialized() { return _map.data && _width > 0 && _height > 0; };

        ///// Operators /////
        // Forward to the Mat_ operators
        cv::Vec6d& operator ()( int y, int x ) { return _map(y,x); };

        ///// Metadata /////
        int width() { return _width; };
        int height() { return _height; };

        ///// Disk IO /////
        void write( boost::filesystem::path path );
        void read( boost::filesystem::path path );

    private:
        int _width, _height;
        cv::Mat_<cv::Vec6d> _map;
    };
}


#endif //VC_PERPIXELMAP_H
