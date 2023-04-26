// OcclusionAccumulation
//   Moving object detection for visual odometry
//   in a dynamic environment based on occlusion accumulation
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
// This software was developed of Jacob.lsx

#ifndef OCCLUSIONACCUMULATION_H_
#define OCCLUSIONACCUMULATION_H_

#include <opencv2/core/core.hpp>

class OcclusionAccumulation {
 public:
  OcclusionAccumulation() = delete;
  OcclusionAccumulation(const cv::Matx33f& K,
                        const cv::Size& size,
                        float depth_unit = 1000.f,
                        float alpha = 0.15f,
                        float beta = 0.225f,
                        int object_threshold = 5e2);

  void movingObjectPrediction();
  void movingObjectDetection(const cv::Mat& depth_cur,
                             const cv::Mat& depth_next,
                             const cv::Matx33f& R,
                             const cv::Vec3f& t,
                             cv::Mat& object_mask);

 private:
  void DepthCompensation(const cv::Matx33f& R, const cv::Vec3f& t);
  void getWarpedImage(cv::InputArray i_ref,
                      cv::InputArray d_ref,
                      cv::InputArray i,
                      cv::InputArray R,
                      cv::InputArray t,
                      cv::InputOutputArray ir,
                      cv::OutputArray residual = cv::noArray(),
                      bool update_warped = true);
  void accumInterpolation(cv::InputOutputArray source_mask,
                          cv::InputOutputArray target_mask,
                          bool update_derive = true);
  std::vector<int> bwConnect(cv::InputArray label,
                             cv::InputArray ref,
                             int number_of_label);

  cv::Mat depth_prev_warped_;
  cv::Mat accumulated_dZdt_;  // A(u)
  cv::Mat predicted_area_;

  cv::Mat background_mask_;   // B(u)
  cv::Mat depth_cur_compensated_;
  cv::Mat depth_next_compensated_;

  cv::Matx33f K_;
  cv::Size size_;
  float depth_unit_;
  float alpha_;
  float beta_;
  int object_threshold_;
};

#endif // OCCLUSIONACCUMULATION_H_
