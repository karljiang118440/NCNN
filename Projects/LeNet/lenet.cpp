#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<fstream>
#include<iostream>

using namespace std;

#include "net.h"

static int prob_net(const cv::Mat& img, std::vector<float>& cls_scores)
{
    ncnn::Net net;
    net.load_param("lenet.param");
    net.load_model("lenet.bin");

    int img_width     = img.cols; // image width
    int img_height    = img.rows; // image height
    int target_width  = 28;       // target resized width
    int target_height = 28;       // target resized height
    printf("test image size: %d %d\n", img_width, img_height);
    
    // lenet使用mnist为灰度图，所以这里ncnn::Mat::PIXEL_BGR2GRAY
    // 如果输入图像为3通道，则ncnn的输入在这里需要将cv::imread读到的BGR转为RGB，使用ncnn::Mat::PIXEL_BGR2RGB
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2GRAY, img_width, img_height, target_width, target_height);
    
    // 输入图片归一化
    const float mean_vals[1] = {0};
    const float norm_vals[1] = {0.00390625f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();

    ex.set_light_mode(true);
    ex.input("data", in);
    ex.extract("prob", out);
    
    cls_scores.resize(out.w);
    // printf("%d\n", out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }
    
    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    int flag = 1;
    cv::Mat m = cv::imread(imagepath, flag);

    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    prob_net(m, cls_scores);
    print_topk(cls_scores, 10);

    return 0;
}
