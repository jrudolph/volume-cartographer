// VC Texture
// Object to store texture information generated by the algorithms in vc_texturing
// Created by Seth Parker on 10/20/15.

#ifndef VC_TEXTURE_H
#define VC_TEXTURE_H

#include <opencv2/opencv.hpp>

#include "common/vc_defines.h"
#include "Metadata.h"
#include "UVMap.h"
#include "PerPixelMap.h"

namespace volcart {
    class Texture {
    public:

        Texture();
        Texture(std::string path);

        // Get metadata
        volcart::Metadata metadata() { return _metadata; };

        std::string     id() { return _metadata.getString("id"); };
        int             width()  { return _width; };
        int             height() { return _height; };
        size_t          numberOfImages()   { return _images.size(); };
        bool            hasImages() { return _images.size() > 0; };
        bool            hasMap()    { return _uvMap.size()  > 0; };

        // Get/Set UV Map
        volcart::UVMap& uvMap(){ return _uvMap; };
        void uvMap(volcart::UVMap uvMap) { _uvMap = uvMap; };

        // Get/Add Texture Image
        cv::Mat getImage(int id) { return _images[id]; };
        void addImage(cv::Mat image);

        // Return the intensity for a Point ID
        double intensity( int point_ID, int image_ID = 0 );

        // Extra Metadata
        void    setMask( cv::Mat m ) { _PerPixelMask = m; };
        cv::Mat getMask() { return _PerPixelMask; };

        void         setMap( PerPixelMap m ) { _PerPixelMapping = m; };
        PerPixelMap  getMap() { return _PerPixelMapping; };

    private:
        volcart::Metadata _metadata;

        boost::filesystem::path _path;
        int _width, _height;
        std::vector<cv::Mat> _images;
        volcart::UVMap _uvMap;

        cv::Mat       _PerPixelMask;
        PerPixelMap   _PerPixelMapping;
    };
} // volcart

#endif //VC_TEXTURE_H
