// OcclusionAccumulation
// -- Moving object detection for visual odometry in a dynamic environment based on occlusion accumulation
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include <dirent.h>

#include <iostream>
#include <utility>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "occlusion_accumulation.h"

int getDir(std::string dir, std::vector<std::string>& files);
int readPose(std::string file,
             std::vector<std::pair<cv::Matx33f, cv::Vec3f>>& poses);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: example_1 [data_path]." << std::endl;
    return 1;
  }

  std::string path = argv[1];
  std::vector<std::string> rgb_files, depth_files;
  std::vector<std::pair<cv::Matx33f, cv::Vec3f>> poses;

  const int num_rgb = getDir(path+"/rgb", rgb_files);
  const int num_depth = getDir(path+"/depth", depth_files);
  const int num_pose = readPose(path+"/pose.txt", poses);

  if (num_rgb != num_depth) {
    std::cerr << "rgb & depth image number must equate." << std::endl;
    return 1;
  }
  if (num_pose < num_depth) {
    std::cerr << "The number of poses needs to be more than the number of images." << std::endl;
    return 1;
  }

  cv::Matx33f K = cv::Matx33f::eye();
  K(0, 0) = 544.3f;
  K(1, 1) = 546.1f;
  K(0, 2) = 326.9f;
  K(1, 2) = 236.1f;

  OcclusionAccumulation occlusion_accumulation(K, cv::Size(640, 480));

  cv::Mat depth_prev = cv::imread(depth_files.at(0), -1);

  double use_time = 0;
  for (int i = 1; i < num_depth; ++i) {
    cv::Mat depth_next = cv::imread(depth_files.at(i), -1);
    cv::Mat rgb = cv::imread(rgb_files.at(i), -1);
    auto& pose = poses.at(i-1);
    cv::Matx33f& R = pose.first;
    cv::Vec3f t = pose.second;

    cv::Mat object_mask;
    auto t1 = std::chrono::steady_clock::now();
    occlusion_accumulation.movingObjectDetection(depth_prev, depth_next, R, t, object_mask);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    use_time += time_used.count();

    depth_prev = depth_next;
    cv::imshow("mask", object_mask);
    cv::waitKey(1);
  }
  std::cout << "occlusion_accumulation.movingObjectDetection mean time cost = "
            << use_time/(num_depth-1) << " seconds. " << std::endl;

  return 0;
}

int getDir(std::string dir, std::vector<std::string>& files) {
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(dir.c_str())) == NULL) {
    return -1;
  }

  while ((dirp = readdir(dp)) != NULL) {
    std::string name = std::string(dirp->d_name);
    if (name != "." && name != "..")
      files.emplace_back(name);
  }
  closedir(dp);

  std::sort(files.begin(), files.end());

  if (dir.at(dir.length()-1) != '/')
    dir += "/";

  for (auto& item : files) {
    if (item.at(0) != '/')
      item = dir + item;
  }

  return files.size();
}

int readPose(std::string file,
             std::vector<std::pair<cv::Matx33f, cv::Vec3f>>& poses) {
  FILE* f_poses = fopen(file.c_str(), "r");
  if (!f_poses) {
    std::cerr << "Cannot be opened the file: " << file << std::endl;
    return -1;
  }

  poses.reserve(200);
  cv::Matx33f R;
  cv::Vec3f t;
  while (fscanf(f_poses, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
                &R(0, 0), &R(0, 1), &R(0, 2), &t[0],
                &R(1, 0), &R(1, 1), &R(1, 2), &t[1],
                &R(2, 0), &R(2, 1), &R(2, 2), &t[2]) != EOF) {
    poses.emplace_back(R, t);
  }
  fclose(f_poses);
  return poses.size();
}
