#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include "../detector.h"
#include <vector>
#include "opencv2/core.hpp"
#include "ncnn/net.h"

class CenterFace : public Detector {
public:
    CenterFace();
    ~CenterFace();
	int LoadModel(const char* root_path);
	int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
    ncnn::Net* centernet_;
    const float scoreThreshold_ = 0.5f;
    const float nmsThreshold_ = 0.5f;
    bool initialized_;
};


#endif // !_FACE_CENTERFACE_H_
