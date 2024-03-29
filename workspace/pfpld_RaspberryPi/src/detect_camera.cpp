#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "pfpld.id.h"



#define _NCNN_PARAM False  // use ncnnoptimize tools to optimize models by karl:20210528

using naspace std;



/*

karl:20210528
 
 1):更改模型，使用 ncnnoptimize tools 优化后的模型

 2） 添加 fps 程序，查看优化后的效果进行比较


karl:20210528


*/


// inline static void stopwatch(bool start ,std::string verb = ""){

// 	static auto startTime = std::chrono::high_resolution_clock::now();


// 	if(start){
// 		startTime = std::chrono::high_resolution_clock::now();		
// 	}
// 	else{
// 		auto endTime = std::chrono::high_resoluiton_clock::now();
// 		std::cout << "Time taken to" << verb << ":"
// 		 << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).cout()
// 		 << "milliseconds"
// 		 <<std::endl;
// 	}

// 	float fps = 1.0 / std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).cout();
// }








/* 增加 对疲劳检测功能 -- add by karl:20210901


1).增加参数


*/


// 计算人疲劳的相关计算公式

#define EYE_AR_THRESH 0.21
#define EYE_AR_CONSEC_FRAMES 1
#define PERCLOS_THRESH 0.4
#define BLINK_FREQ_MINUTE 9


double calcTwoNormIsEuclid(cv::Point p1, cv::Point p2);
double eyeAspectRatio(std::vector<cv::Point> eye);
int round_double(double number);


#define ScaleKarl 4

string initstate = "STATE: ";
string iniDriverState = "No Face Detected !"
string deteDriverState = "Face Detected !"




bool flag_one = true;
bool flag_two = true;
bool flag_three  = true;
bool flag_four = true;
bool flag_big = true;

int karl_flag;

string tmp_display_fatigue;

int test_e5 =0;

char fataguePara[4][200] = {{"%s ; you have fagigued !"},
{"%s ; you have fatigued !"},
{"%s ; you have fatigued !"},
{"%s ; you have fatigued !"} };

int array2[4] = {0,1,2,3};
















int main() {
    extern float pixel_mean[3];
    extern float pixel_std[3];


	std::string param_path =  "../models/pfpld/scrfd_500m-opt2.param";
	std::string bin_path = "../models/pfpld/scrfd_500m-opt2.bin";
	std::string pfpld_path = "../models/pfpld/pfpld.ncnnmodel";





	//  add fps karl:20210528

	char string[10];
	double t =0;
	double fps;
	



	ncnn::Net _net, pfpld_net;
	_net.load_param(param_path.data());
	_net.load_model(bin_path.data());

	FILE *fp = fopen(pfpld_path.c_str(), "rb");
	if (fp != nullptr) {
		pfpld_net.load_param_bin(fp);
		pfpld_net.load_model(fp);
		fclose(fp);
	}
            

	cv::Mat img ;
	
	
	cv::VideoCapture cap(0);

    if(!img.data)
    	printf("load error");




	/* 增加相关的参数说明列表
	
	
	
	*/

	int ear_count =0;
	int total_eye_blink = 0;
	int count_frame =0;
	double preclos;
	double blinkFreq;
	double clac_sum_time;
	int t30s =0;
	int unitTimeEyeCloseFrame_n = 0;
	int fatigue =0;
	int ear_close_state =0;
	int gCount =0;
	int criticalValue =0;







	while(1){

	// add stopwatch() by karl:210210528

	// stopwatch(true);



	//  add fps karl:20210528

	t = (double)cv::getTickCount();

	// printf("line114 \n");

	cap >> img;

	// cv::imshow("img", img);

	ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, img.cols, img.rows);

//    cv::resize(img, img, cv::Size(300, 300));

    input.substract_mean_normalize(pixel_mean, pixel_std);
	ncnn::Extractor _extractor = _net.create_extractor();
	_extractor.input("data", input);


    std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        int stride = _feat_stride_fpn[i];
        ac[i].Init(stride, anchor_cfg[stride], false);
    }

    std::vector<Anchor> proposals;
    proposals.clear();

    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    	ncnn::Mat cls;
    	ncnn::Mat reg;
    	ncnn::Mat pts;

        // get blob output
        char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
        char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
        char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
        _extractor.extract(clsname, cls);
        _extractor.extract(regname, reg);
        _extractor.extract(ptsname, pts);

        // printf("cls %d %d %d\n", cls.c, cls.h, cls.w);
        // printf("reg %d %d %d\n", reg.c, reg.h, reg.w);
        // printf("pts %d %d %d\n", pts.c, pts.h, pts.w);

        ac[i].FilterAnchor(cls, reg, pts, proposals);

        // printf("stride %d, res size %d\n", _feat_stride_fpn[i], proposals.size());

        for (int r = 0; r < proposals.size(); ++r) {
            proposals[r].print();
        }
    }

    // nms
    std::vector<Anchor> result;
    nms_cpu(proposals, nms_threshold, result);

    // printf("final result %d\n", result.size());
    for(int i = 0; i < result.size(); i ++)
    {
        cv::rectangle (img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(255, 255, 0), 2, 8, 0);
//        for (int j = 0; j < result[i].pts.size(); ++j) {
//        	cv::circle(img, cv::Point((int)result[i].pts[j].x, (int)result[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
//        }
        int x1 = (int)result[i].finalbox.x;
        int y1 = (int)result[i].finalbox.y;
        int x2 = (int)result[i].finalbox.width;
        int y2 = (int)result[i].finalbox.height;
		int height = img.rows;
		int width = img.cols;
		int channel = img.channels();
        int w = x2 - x1 + 1;
        int h = y2 - y1 + 1;

		int size_w = (int)(MAX(w, h)*0.9);
		int size_h = (int)(MAX(w, h)*0.9);
		int cx = x1 + w / 2;
		int cy = y1 + h / 2;
		x1 = cx - size_w / 2;
		x2 = x1 + size_w;
		y1 = cy - (int)(size_h * 0.4);
		y2 = y1 + size_h;
		
		int left = 0;
		int top = 0;
		int bottom = 0;
		int right = 0;
		if(x1 < 0)
            left = -x1;
		if (y1 < 0)
            top = -y1;
		if (x1 >= width)
            right = x2 - width;
		if (y1 >= height)
            bottom = y2 - height;
		
		x1 = MAX(0, x1);
		y1 = MAX(0, y1);
		
		x2 = MIN(width, x2);
		y2 = MIN(height, y2);
		
		cv::Mat face_img = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
		cv::copyMakeBorder(face_img, face_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
		

		cv::resize(face_img, face_img, cv::Size(112, 112));
		
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(
		(unsigned char*)face_img.data,
		ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
		float mean_vals[3] = {0.0, 0.0, 0.0};
        	float norm_vals[3] = {1 / (float)255.0, 1 / (float)255.0, 1 / (float)255.0};
		ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
		
		ncnn::Extractor pfpld_ex = pfpld_net.create_extractor();
		ncnn::Mat pose, landms;
        std::vector<float> angles;
        std::vector<float> landmarks;
		pfpld_ex.input(pfpld_param_id::BLOB_input, ncnn_img);
		pfpld_ex.extract(pfpld_param_id::BLOB_pose, pose);
		pfpld_ex.extract(pfpld_param_id::BLOB_landms, landms);
		for (int j=0; j<pose.w; j++){
            float tmp_angle = pose[j] * 180.0 / CV_PI;
            angles.push_back(tmp_angle);
		}
		
		// for (int j=0; j<landms.w / 2; j++)
		
		for ( int j=60; j<95; j++) // 仅显示人眼睛和嘴巴的关键点部位 --add by karl:20210901
		{
            float tmp_x = landms[2 * j] * size_w + x1 - left;
            float tmp_y = landms[2 * j + 1] * size_h + y1 -bottom;
            landmarks.push_back(tmp_x);
            landmarks.push_back(tmp_y);
            cv::circle(img, cv::Point((int)tmp_x, (int)tmp_y), 1, cv::Scalar(0,255,0), 1);
			printf(" landms.w= %d \n",landms.w);
		}
		std::cout<<angles[0]<<"  "<<angles[1]<<"  "<<angles[2]<<std::endl;
		plot_pose_cube(img, angles[0], angles[1], angles[2], (int)result[i].pts[2].x, (int)result[i].pts[2].y, w / 2);
    }

    // result[0].print();

    cv::imshow("img", img);


	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	fps = 1.0 /t;

	sprintf(string, "%.2f",fps);

	std::string fpsString("FPS: ");
	fpsString += string;

	printf("%f \n", fps);

	cv::putText(img,fpsString,cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(100,0,0));


    if(cv::waitKey(20) > 0)
        break;



}


}




double eyeAspectRatio(std::vector<cv::Point> eye){


	double short_axis_A = calcTwoNormIsEuclid(eye[5],eye[1]);
	double short_axis_B = calcTwoN 
}

double calcTwoNormIsEuclid(cv::Point p1 ,cv::Point p2){

	double dist;
	dist = sqrt(pow(p2.x - p1.x),2) + pow((p2.y - p1.y),2);
	return  dist;
}




