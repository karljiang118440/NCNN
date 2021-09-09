#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "pfpld.id.h"



#define _NCNN_PARAM False  // use ncnnoptimize tools to optimize models by karl:20210528

using namespace std;
using namespace cv;



#include<algorithm>
#include<iostream>
#include<math.h>

// list of source code updata:

/*

karl:20210528
 
 1):更改模型，使用 ncnnoptimize tools 优化后的模型

 2） 添加 fps 程序，查看优化后的效果进行比较


karl:20210528



karl:20210907


 1): 将疲劳模型计算在内


*/






/* 增加 对疲劳检测功能 -- add by karl:20210901


1).增加参数


*/


// 计算人疲劳的相关计算公式

#define EYE_AR_THRESH 0.21
#define EYE_AR_CONSEC_FRAMES 1
#define PERCLOS_THRESH 0.4
#define BLINK_FREQ_MINUTE 9

#define ScaleKarl 4


double calcTwoNormIsEuclid(cv::Point2f p1 ,cv::Point2f p2);

double eyeAspectRatio_98landmarks(std::vector<cv::Point2f> eye1);

int round_double(double number);




// string initstate = "STATE: ";
string Initstate_dms = "STATE: ";
string iniDriverState = "No Face Detected !";
string deteDriverState = "Face Detected !";




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




/* add parameters of Eyes --by karl:20210907

*/

	std::vector<cv::Point2f>   leftEye,rightEye;
	std::vector<cv::Point2f>   Eye; // 只需要定义 Eye 即可，不需要进行其他的操作。 --by karl:20210909


//************************* end *********************//




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


		printf("line189 \n");




	while(1){

	// add stopwatch() by karl:210210528

	// stopwatch(true);



	//  add fps karl:20210528

	t = (double)cv::getTickCount();

	// printf("line114 \n");

	cap >> img;

	cv::imshow("img", img);

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

			// printf("tmp_x = %f \n",tmp_x);
			// printf("tmp_y = %f \n",tmp_y);


			// // cv::Point tmp_point = cv::Point(tmp_x,tmp_y);
			// cv::Point2f tmp_point = cv::Point2f(300.5,100.5);
			// printf("x , y = %.3f , %.3f \n",tmp_point.x ,tmp_point.y);


			// leftEye.push_back(cv::Point2f(tmp_x,tmp_y));
			// printf(" line331:lefteye.x = %f \n",leftEye[1].x);
			// printf(" line332:lefteye.y = %f \n",leftEye[1].y);


	//1). extract left eye and right eye -- add by karl:20210907
			// 提取Eye 信息  -- add by karl:20210909
			Eye.push_back(cv::Point2f(tmp_x,tmp_y));
			printf("x , y = %.3f , %.3f \n",Eye[1].x ,Eye[1].y);
			printf("x , y = %.3f , %.3f \n",Eye[7].x ,Eye[7].y);



			// if(i >= 60  && i <= 67 ){
			// 	tmp_x = landms[2 * i] * size_w + x1 - left;
			// 	tmp_y = landms[2 * i + 1] * size_h + y1 -bottom;


			// 	printf("tmp_y:line336 = %f \n",tmp_y);

			// 	leftEye.push_back(cv::Point(tmp_x,tmp_y));

				
			// }

			// if(i >= 68  && i <= 75 ){
			// 	tmp_x = landms[2 * i] * size_w + x1 - left;
			// 	tmp_y = landms[2 * i + 1] * size_h + y1 -bottom;				
			// 	rightEye.push_back(cv::Point(tmp_x,tmp_y));
			// }


			// printf("rightEye = %f",rightEye[70].x);
			// printf("rightEye:line351 = %f \n",rightEye[1]);
			// printf("rightEye:line352 = %f \n",rightEye[70]);
			// printf("rightEye:line353 = %f \n",rightEye);
	//*******************end ********************//

		}


	// 2). calculate eye aspect ratio -- add by karl:20210907

		double leftEyeEar = eyeAspectRatio_98landmarks(Eye);
		double rightEyeEar = eyeAspectRatio_98landmarks(Eye);

		double avg_Ear = 0.5*(leftEyeEar + rightEyeEar);

		printf("avg_Ear = %f \n", avg_Ear);

		FILE *file_in = fopen("leftEyeEar.txt","a");
		fprintf(file_in,"leftEyeEar: %.2f , rightEyeEar: %.2f,ratio: .2f \n",leftEyeEar,rightEyeEar,avg_Ear);
		fclose(file_in);


		std::cout<<angles[0]<<"  "<<angles[1]<<"  "<<angles[2]<<std::endl;
		plot_pose_cube(img, angles[0], angles[1], angles[2], (int)result[i].pts[2].x, (int)result[i].pts[2].y, w / 2);
    }

    // result[0].print();



    cv::imshow("img", img);

	// t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	// fps = 1.0 /t;

	// sprintf(string, "%.2f",fps);

	// std::string fpsString("FPS: ");
	// fpsString += string;

	// printf("fps = %f \n", fps);

	// cv::putText(img,fpsString,cv::Point(100,100),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(100,0,0));


    if(cv::waitKey(20) > 0)
        break;



}





}



















// double eyeAspectRatio(std::vector<cv::Point> eye){



// 	double short_axis_A = calcTwoNormIsEuclid(eye[5] , eye[1]);

// 	double short_axis_B = calcTwoNormIsEuclid(eye[4] , eye[2]);
// 	double long_axis_C = calcTwoNormIsEuclid(eye[3],eye[0]);
// 	double calc_ear = (short_axis_A + short_axis_B) / (2.0 * long_axis_C);
// 	return calc_ear;

// }



double eyeAspectRatio_98landmarks(std::vector<cv::Point2f> eye1){




			printf(" line440 \n");

	// double short_axis_B = calcTwoNormIsEuclid(eye[6].y , eye[2].y);
	// double short_axis_C = calcTwoNormIsEuclid(eye[5].y , eye[3].y);
	// double long_axis_D = calcTwoNormIsEuclid(eye[4].x,  eye[0].x);


	double short_axis_A = calcTwoNormIsEuclid(eye1[7] , eye1[1]);
	double short_axis_B = calcTwoNormIsEuclid(eye1[6] , eye1[2]);
	double short_axis_C = calcTwoNormIsEuclid(eye1[5] , eye1[3]);
	double long_axis_D = calcTwoNormIsEuclid(eye1[4],  eye1[0]);

	double calc_ear = (short_axis_A + short_axis_B + short_axis_C) / (3.0 * long_axis_D);
	return calc_ear;

}

double calcTwoNormIsEuclid(cv::Point2f p1 ,cv::Point2f p2){


	printf(" line453 \n");
	double dist;


	// int x = p2.x - p1.x;
	// int y = p2.y - p1.y;


	// dist = sqrt(pow(x,2) + pow(y,2));


	dist = sqrt(pow(int((p2.x - p1.x)),2) + pow(int((p2.y - p1.y)),2));	

			printf(" line457 \n");


	// dist = sqrt(pow(2,2) + pow(2,2));	


	return  dist;
}






// string doubleTwoRound(double dVal){

// 	char buf_time[10];
// 	sprintf(buf_time, "%.2f",dVal);
// 	stringstream ss;
// 	ss << buf_time;
// 	return ss.str();
// }

// string getCurrentTime()
// {
// 	time_t t = time(NULL);
// 	char ch[64] = {0};
// }

// cv::Mat scrollScreen(int fatigue,cv::Mat & temp){

// 	char fatigue_path[200];


// 	if(fatigue <= 4)
// 	{
// 		if (fatigue == 1){

// 			if(flag_one == true){

// 				sprintf(fatigue_path,fataguePara[0],getCurrentTime().c_str());
// 				fa.push_back(fatigue_path);
// 				flag_one = false;	
// 			}
// 			cv::putText(temp,fa[0],point_arr[0],CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(0,255,0),2);
// 		}

// 		else if(fatigue ==2){
// 			if (flag_two == true){
// 				sprintf(fatigue_path,fataguePara[1],getCurrentTime().c_str());
// 				fa.push_back(fatigue_path);
// 				flag_two = false;
// 			}

// 			for(unsigned int i=0;i < fa.size();i++){

// 				if( i < fa.size() -1){

// 					cv::putText(temp,fa[i],point_arr[i],CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(125,125,125),2);

// 				}

// 				else {
// 					cv::putText(temp,fa[i],point_arr[i],CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(0,255,0),2);
// 				}
// 			}
// 		}

// 		else if (fatigue ==3)
// 		{
// 			if(flag_three == true)
// 			{
// 				sprintf(fatigue_path,fataguePara[2],getCurrentTime().c_str());
// 				fa.push_back(fatigue_path);
// 				flag_three =false;
// 			}

// 			for (unsigned int i=0;i< fa.size();i++)
// 			{
// 				if (i < fa.size() -1)
// 				{
// 					cv::putText(temp, fa[i], point_arr[i], CV_FONT_HERSHEY_SIMPLEX, FONT_SIZE_DIS, cv::Scalar(125, 125, 125), 2);
// 				}
// 				else
// 				{
// 					cv::putText(temp, fa[i], point_arr[i], CV_FONT_HERSHEY_SIMPLEX, FONT_SIZE_DIS, cv::Scalar(0, 255, 0), 2);
// 				}
// 			}
// 		}

// 		else if(fatigue == 4) 
// 		{
// 			if (flag_four == true )
// 			{
// 				sprintf(fatigue_path, fataguePara[3],getCurrentTime().c_str());
// 				fa.push_back(fatigue_path);
// 				flag_four = false;
// 			}

// 			for (unsigned int i=0;i < fa.size();i ++)
// 			{
// 				if ( i < fa.size() -1)
// 				{
// 					cv::putText(temp,fa[i],point_arr[i],CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(125,125,125),2);
// 				}
// 				else 
// 				{
// 					cv::putText(temp,fa[i],point_arr[i],CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(0,255,0),2);

// 				}
// 			}
// 		}

// 		if (karl_flag!=  fatigue)
// 		{
// 			flag_big = true;
// 			tmp_display_fatigue = "";

// 		}

// 		if(flag_big == true)
// 		{
// 			for(unsigned int i=0;i <4;i++)
// 			{
// 				array2[i] = (array2[i] + 1) %4;
// 			}

// 			cout << "sizeof(array2) =" << sizeof(array2) << endl;

// 			for(unsigned int i =0; i < 3; i++)
// 			{
// 				tmp_display_fatigue += fa[array2[i]];
// 				tmp_display_fatigue += "\n";
// 			}

// 			sprintf( fatigue_path,"%s ; You have fatigued ! \n" , getCurrentTime().c_str());
// 			tmp_display_fatigue + = fatigue_path;

// 			fa.erase(fa.begin());
// 			fa.push_back(fatigue_path);

// 			flag_big = false;
// 			karl_flag = fatigue;

// 			for (unsigned int i=0;i<fa.size();i++)
// 			{
// 				cout << "fa[" << i << "] = " << fa[i] << endl;
// 			}

// 			cout << "come in : %d" << test_e5++ <<endl;
// 			cout << "f = " << tmp_display_fatigue << endl;

// 		}



// 		std::vector<string> chara_string_spli = characterStingSplit(tmp_display_fatigue);
// 		for (unsigned int i=0;i < chara_string_spli.size();i++)
// 		{
// 			if(i < chara_string_spli.size() -1)
// 			{
// 				cv::putText(temp,chara_string_spli[i],cv::Point(point_arr[0].x, point_arr[0].y + i* SCREEN_LINE_SEG),CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(125,125,125),2);
// 			}
// 			else
// 			{
// 				cv::putText(temp,chara_string_spli[i],cv::Point(point_arr[0].x,point_arr[0].y + i* SCREEN_LINE_SEG),CV_FONT_HERSHEY_SIMPLEX,FONT_SIZE_DIS,cv::Scalar(0,125,0),2);
// 			}
			
// 		}

		
// 	}

// 	return temp;


// }


// std::vector<string> characterStingSplit(string pend_string)
// {

// 	std::vector<string> stor_split_string;

// 	char *dd_p = const_cast<char*> (pend_string.c_str());
// 	const char * split ="\n";
// 	char *p;

// 	p = strtok(dd_p , split);
// 	while(p! = NULL)
// 	{
// 		stor_split_string.push_back(p);
// 		p = strtok(NULL,split);
// 	}
// 	return stor_split_string;
// }






























