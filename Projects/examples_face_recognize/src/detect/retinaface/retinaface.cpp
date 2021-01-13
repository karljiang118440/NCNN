#include "retinaface.h"
#include <iostream>

RetinaFace::RetinaFace() :
	retina_net_(new ncnn::Net()),
	initialized_(false) {}

RetinaFace::~RetinaFace() {
	retina_net_->clear();
}

int RetinaFace::LoadModel(const char * root_path) {
	std::string fd_param = std::string(root_path) + "/fd.param";
	std::string fd_bin = std::string(root_path) + "/fd.bin";
	if (retina_net_->load_param(fd_param.c_str()) == -1 ||
		retina_net_->load_model(fd_bin.c_str()) == -1) {
		std::cout << "load face detect model failed." << std::endl;
		return 10000;
	}

	// generate anchors
	for (int i = 0; i < 3; ++i) {
		ANCHORS anchors;
		if (0 == i) {
			GenerateAnchors(16, { 1.0f }, { 32, 16 }, &anchors);
		}
		else if (1 == i) {
			GenerateAnchors(16, { 1.0f }, { 8, 4 }, &anchors);
		}
		else {
			GenerateAnchors(16, { 1.0f }, { 2, 1 }, &anchors);
		}
		anchors_generated_.push_back(anchors);
	}
	initialized_ = true;

	return 0;
}

int RetinaFace::Detect(const cv::Mat & img_src,
	std::vector<FaceInfo>* faces) {
	std::cout << "start face detect." << std::endl;
	faces->clear();
	if (!initialized_) {
		std::cout << "retinaface detector model uninitialized." << std::endl;
		return 10000;
	}
	if (img_src.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}
	cv::Mat img_cpy = img_src.clone();
	int img_width = img_cpy.cols;
	int img_height = img_cpy.rows;
	float factor_x = img_width / 300.0f;
	float factor_y = img_height / 300.0f;
	ncnn::Extractor ex = retina_net_->create_extractor();
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_width, img_height, 300, 300);
	ex.input("data", in);

	std::vector<float> scores;
	std::vector<FaceInfo> faces_tmp;
	for (int i = 0; i < 3; ++i) {
		std::string class_layer_name = "face_rpn_cls_prob_reshape_stride" + std::to_string(RPNs[i]);
		std::string bbox_layer_name = "face_rpn_bbox_pred_stride" + std::to_string(RPNs[i]);
		std::string landmark_layer_name = "face_rpn_landmark_pred_stride" + std::to_string(RPNs[i]);

		ncnn::Mat class_mat, bbox_mat, landmark_mat;
		ex.extract(class_layer_name.c_str(), class_mat);
		ex.extract(bbox_layer_name.c_str(), bbox_mat);
		ex.extract(landmark_layer_name.c_str(), landmark_mat);

		ANCHORS anchors = anchors_generated_.at(i);
		int width = class_mat.w;
		int height = class_mat.h;
		int anchor_num = static_cast<int>(anchors.size());
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				int index = h * width + w;
				for (int a = 0; a < anchor_num; ++a) {
					float score = class_mat.channel(anchor_num + a)[index];
					if (score < 0.8) {
						continue;
					}
					scores.push_back(score);
					cv::Rect box = cv::Rect(w * RPNs[i] + anchors[a].x,
						h * RPNs[i] + anchors[a].y,
						anchors[a].width,
						anchors[a].height);

					float delta_x = bbox_mat.channel(a * 4 + 0)[index];
					float delta_y = bbox_mat.channel(a * 4 + 1)[index];
					float delta_w = bbox_mat.channel(a * 4 + 2)[index];
					float delta_h = bbox_mat.channel(a * 4 + 3)[index];
					cv::Point2f center = cv::Point2f(box.x + box.width * 0.5f,
						box.y + box.height * 0.5f);
					center.x = center.x + delta_x * box.width;
					center.y = center.y + delta_y * box.height;
					float curr_width = std::exp(delta_w) * (box.width + 1);
					float curr_height = std::exp(delta_h) * (box.height + 1);
					cv::Rect curr_box = cv::Rect(
						center.x - curr_width * 0.5f,
						center.y - curr_height * 0.5f,
						curr_width,
						curr_height);
					curr_box.x = MAX(curr_box.x * factor_x, 0);
					curr_box.y = MAX(curr_box.y * factor_y, 0);
					curr_box.width = MIN(img_width - curr_box.x, curr_box.width * factor_x);
					curr_box.height = MIN(img_height - curr_box.y, curr_box.height * factor_y);

					FaceInfo face(curr_box, score);
					faces_tmp.push_back(face);
				}
			}
		}
	}

	std::sort(faces_tmp.begin(), faces_tmp.end(), [](FaceInfo face1, FaceInfo face2) { return face1.score_ < face2.score_; });
	NMS(faces_tmp, faces, iouThreshold_);
	std::cout << faces->size() << " faces detected." << std::endl;

	std::cout << "end face detect." << std::endl;
	return 0;
}
