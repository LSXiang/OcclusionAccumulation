// OcclusionAccumulation
//   Moving object detection for visual odometry
//   in a dynamic environment based on occlusion accumulation
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
// This software was developed of Jacob.lsx

#include <iostream>
#include <fstream>
#include <utility>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "occlusion_accumulation.h"

void loadImages(const std::string& data_path,
                std::vector<std::string>& rgb_img,
                std::vector<std::string>& depth_l_img,
                std::vector<std::pair<cv::Matx33f, cv::Vec3f>>& poses);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: example_2 [data_path]." << std::endl;
    return 1;
  }

  std::string path = argv[1];
  std::vector<std::string> rgb_files, depth_files;
  std::vector<std::pair<cv::Matx33f, cv::Vec3f>> poses;

  loadImages(path, rgb_files, depth_files, poses);

  cv::Matx33f K = cv::Matx33f::eye();
  K(0, 0) = 378.3790588378906;
  K(1, 1) = 378.02239990234375;
  K(0, 2) = 318.745849609375;
  K(1, 2) = 244.57534790039062;

  const auto num_depth = depth_files.size();

  OcclusionAccumulation occlusion_accumulation(K, cv::Size(640, 480), 1000.f, 0.15f, 0.225f, 1000, 25);

  cv::Mat depth_prev = cv::imread(depth_files.at(0), -1);
  auto last_pose = poses.at(0);

  double use_time = 0;
  for (int i = 1; i < num_depth; ++i) {
    cv::Mat depth_next = cv::imread(depth_files.at(i), -1);
    cv::Mat rgb = cv::imread(rgb_files.at(i), -1);
    auto& pose = poses.at(i);
    cv::Matx33f R = pose.first.t()*last_pose.first;
    cv::Vec3f t = pose.first.t() * (last_pose.second - pose.second);

    cv::Mat object_mask;
    auto t1 = std::chrono::steady_clock::now();
    object_mask = occlusion_accumulation.movingObjectDetection(depth_prev, depth_next, R, t);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    use_time += time_used.count();

    depth_prev = depth_next;
    last_pose = pose;

    cv::Mat mask(rgb.size(), rgb.type(), cv::Scalar(0, 0, 0));
    mask.setTo(cv::Scalar(127, 0, 127), object_mask);
    cv::imshow("Moving object detection", mask + rgb);
    cv::waitKey(1);
  }
  std::cout << "occlusion_accumulation.movingObjectDetection mean time cost = "
            << use_time/(num_depth-1) << " seconds. " << std::endl;

  return 0;
}

void loadImages(const std::string& data_path,
                std::vector<std::string>& rgb_img,
                std::vector<std::string>& depth_l_img,
                std::vector<std::pair<cv::Matx33f, cv::Vec3f>>& poses) {
  std::string times_path = data_path + "/time.txt";
  std::ifstream f_time;
  f_time.open(times_path);
  if (!f_time.is_open()) {
    std::cerr << "Cannot be opened the file: " << times_path << std::endl;
    return;
  }

  rgb_img.reserve(3000);
  depth_l_img.reserve(3000);

  int cnt = 0;
  while (!f_time.eof()) {
    std::string s;
    std::getline(f_time, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      if (cnt++ % 2 == 0) {
        rgb_img.push_back(data_path + "/image_rgb/" + ss.str() + ".png");
      } else {
        depth_l_img.push_back(data_path + "/image_depth_alignrgb/" + ss.str() + ".png");
      }
    }
  }
  f_time.close();

  std::string poses_path = data_path + "/CameraTrajectory.txt";
  FILE* f_poses = fopen(poses_path.c_str(), "r");
  if (!f_poses) {
    std::cerr << "Cannot be opened the file: " << poses_path << std::endl;
    return;
  }
  poses.reserve(3000);
  std::array<double, 7> pose = {0};
  cv::Matx33f R;
  cv::Vec3f t;
  while (fscanf(f_poses, "%f %f %f %f %f %f %f %f %f %f %f %f",
                &R(0, 0), &R(0, 1), &R(0, 2), &t[0],
                &R(1, 0), &R(1, 1), &R(1, 2), &t[1],
                &R(2, 0), &R(2, 1), &R(2, 2), &t[2]) != EOF) {
    poses.emplace_back(R, t);
  }
  fclose(f_poses);
  std::cout << "pose size: " << poses.size() << std::endl;
}
