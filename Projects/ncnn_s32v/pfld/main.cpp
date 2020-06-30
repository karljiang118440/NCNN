/*****************************************************************************
*
* Freescale Confidential Proprietary
*
* Copyright (c) 2014 Freescale Semiconductor;
* All Rights Reserved
*
*****************************************************************************
*
* THIS SOFTWARE IS PROVIDED BY FREESCALE "AS IS" AND ANY EXPRESSED OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL FREESCALE OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*
****************************************************************************/
#include <stdlib.h>
#include <stdio.h>



#define NCNN true


#if NCNN

#include <map>
#include <vector>  
#include <algorithm>  
#include <functional>  
#include <cstdlib> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <ncnn/net.h>
#include <net.h>


#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "platform.h"
#include "net.h"

#endif // NCNN_VULKAN
 


using namespace std;
using namespace cv;
using namespace ncnn;


static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;
    //net squeezenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
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

















/*======================================================================================*/
/* Main function									*/
/*======================================================================================*/
int main(int, char**)
{
  // Constants
  int i = 0;

  printf("Hello World...\n");

  while(i < 10)
  {
    printf("Iteration %i\n", i++);
  }

 // return 0;



//double Time = (double)cvGetTickCount();



#if 0

	std::string imagepath = "./test.jpg";
	cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	if (m.empty())
	{
		std::cout << "cv::imread " << imagepath << " failed\n" << std::endl;
		return -1;
	}


	std::vector<float> cls_scores;
	detect_squeezenet(m, cls_scores);

	print_topk(cls_scores, 3);
	//cv::waitKey(0);


//  算法过程
Time = (double)cvGetTickCount() - Time ;

printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//毫秒

//printf( "run time = %gs\n", Time /(cvGetTickFrequency()*1000000) );//秒
    

	//return 0;


printf("ncnn success \n");


#endif




#if 1

double Time = (double)cvGetTickCount();

ncnn::Net pfld;
pfld.load_param("pfld-sim.param");
pfld.load_model("pfld-sim.bin");

std::string imagepath = "./1.jpeg";

cv::Mat img = cv::imread(imagepath, 1);
ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 112, 112);
const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
in.substract_mean_normalize(0, norm_vals);



ncnn::Extractor ex = pfld.create_extractor();
ex.input("input_1", in);
ncnn::Mat out;
ex.extract("415", out);




Time = (double)cvGetTickCount() - Time ;

printf( "run time = %gms\n", Time /(cvGetTickFrequency()*1000) );//毫秒


#endif 





}
