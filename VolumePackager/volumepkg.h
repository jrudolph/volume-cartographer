#include <opencv2/opencv.hpp>
#include "volumepkgcfg.h"
#include <stdlib.h>

class VolumePkg {
public:
	VolumePkg(std::string);
	int getNumberOfSlices();
	cv::Mat getSliceAtIndex(int);
private:
	VolumePkgCfg config;
	std::string location;
	int getNumberOfSliceCharacters();
};