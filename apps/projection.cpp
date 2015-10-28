// projection.cpp
// Seth Parker 10/2015
// Project the mesh onto each slices in order to check the quality of segmentation
#include <stdio.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

#include "vc_defines.h"
#include "vc_datatypes.h"

#include "io/ply2itk.h"
#include "volumepkg.h"

#define RED cv::Vec3b( 0, 0, 255 );

int main( int argc, char *argv[] ) {
    printf("Running tool: vc_projection\n");
    std::cout << std::endl;
    if (argc < 4) {
        printf("Usage: vc_projection volpkg seg-id output-dir\n");
        exit(-1);
    }

    // Load the volume package
    VolumePkg volpkg(argv[1]);
    volpkg.setActiveSegmentation(argv[2]);
    std::string outputDir = argv[3];

    // Load the mesh
    int width = -1;
    int height = -1;
    VC_MeshType::Pointer mesh = VC_MeshType::New();
    volcart::io::ply2itkmesh(volpkg.getMeshPath(), mesh, width, height);

    //PNG Compression params
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    // Sort the points by z-index
    std::map< int, std::vector<int> > z_map;

    for ( VC_PointsInMeshIterator point = mesh->GetPoints()->Begin(); point != mesh->GetPoints()->End(); ++point ) {

        // skip null points
        if (point->Value()[VC_INDEX_Z] == -1) continue;

        // get the z-index for this point (floored int)
        int this_z = (int) point->Value()[VC_INDEX_Z];

        // add point id to the vector for this z-index
        // make a vector for this z-index if it doesn't exist in the map
        auto lookup_z = z_map.find(this_z);
        if ( lookup_z != z_map.end() ) {
            lookup_z->second.push_back(point->Index());
        } else {
            std::vector<int> z_vector;
            z_vector.push_back(point->Index());
            z_map.insert( {this_z, z_vector} );
        }

    }

    // Iterate over each z-index and generate a projected slice image
    for ( auto z_id = z_map.begin(); z_id != z_map.end(); ++z_id ) {
        std::cout << "Projecting slice " + std::to_string(z_id->first) + "\r" << std::flush;
        // get the slice image and cvt to CV_8UC3
        // .clone() to make sure we don't modify the cached version
        cv::Mat slice = volpkg.getSliceData(z_id->first).clone();
        slice.convertTo( slice, CV_8U, 0.00390625);
        cv::cvtColor( slice, slice, CV_GRAY2BGR );

        // Iterate over the points for this z-index and project the points
        for ( auto p_id = z_id->second.begin(); p_id != z_id->second.end(); ++p_id ) {

            VC_PointType point = mesh->GetPoint(*p_id);
            int x = cvRound(point[VC_INDEX_X]);
            int y = cvRound(point[VC_INDEX_Y]);

            slice.at< cv::Vec3b >(y, x) = RED;
        }

        // Generate an output path from the z-index
        std::stringstream outputPath;
        outputPath << boost::format( outputDir + "/proj%04i.png") % z_id->first;
        cv::imwrite(outputPath.str(), slice, compression_params);
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}