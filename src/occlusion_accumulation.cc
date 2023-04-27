// OcclusionAccumulation
//   Moving object detection for visual odometry
//   in a dynamic environment based on occlusion accumulation
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
// This software was developed of Jacob.lsx

#include "occlusion_accumulation.h"

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


OcclusionAccumulation::OcclusionAccumulation(const cv::Matx33f& K,
                                             const cv::Size& size,
                                             float depth_unit,
                                             float alpha ,
                                             float beta,
                                             int object_threshold)
    : K_(K), size_(size), depth_unit_(depth_unit),
      alpha_(alpha), beta_(beta), object_threshold_(object_threshold) {
  depth_prev_warped_ = cv::Mat::zeros(size_, CV_32F);
  accumulated_dZdt_ = cv::Mat::zeros(size_, CV_32F);
  predicted_area_ = cv::Mat::zeros(size_, CV_8U);

  background_mask_ = cv::Mat::ones(size_, CV_8U);
  depth_cur_compensated_ = cv::Mat::ones(size_, CV_32F);
  depth_next_compensated_ = cv::Mat::ones(size_, CV_32F);
}

void OcclusionAccumulation::movingObjectPrediction() {

}

#include <fstream>

void OcclusionAccumulation::movingObjectDetection(const cv::Mat& depth_cur,
                                                  const cv::Mat& depth_next,
                                                  const cv::Matx33f& R,
                                                  const cv::Vec3f& t,
                                                  cv::Mat& object_mask) {
  depth_cur.convertTo(depth_cur_compensated_, CV_32F, 1.f/depth_unit_);
  depth_next.convertTo(depth_next_compensated_, CV_32F, 1.f/depth_unit_);

  // Depth compensation
  DepthCompensation(R, t);

  // Compute the occlusion Dzdt
  cv::Mat depth_cur_warped, dZdt;
  getWarpedImage(depth_next_compensated_, depth_cur_compensated_,
                 depth_cur_compensated_, R.t(), -R.t()*t,
                 depth_cur_warped, dZdt, false);

  // Warp the occlusion accumulation map, newly discovered area
  getWarpedImage(depth_next_compensated_, depth_cur_compensated_,
                 accumulated_dZdt_, R.t(), -R.t()*t,
                 accumulated_dZdt_, cv::noArray(), false);
  getWarpedImage(depth_next_compensated_, depth_cur_compensated_,
                 predicted_area_, R.t(), -R.t()*t,
                 predicted_area_, cv::noArray(), false);

  // Accumulate the occlusion
  accumulated_dZdt_ = accumulated_dZdt_ + dZdt;

  // set threshold
  auto tau_alpha = alpha_ * depth_next_compensated_;
  auto tau_beta = beta_ * depth_next_compensated_;

  // Truncate error values
  // Equation 5
  accumulated_dZdt_.setTo(0.f, accumulated_dZdt_ <= tau_alpha);
  // Equation 6
  accumulated_dZdt_.setTo(0.f, dZdt <= -tau_beta);

  // Occlusion prediction on newly discovered area
  cv::Mat new_discovered_area(depth_next_compensated_.size(),
                              CV_8U,
                              cv::Scalar(0));
  new_discovered_area.setTo(1,
      (depth_cur_warped == 0.f) - (depth_next_compensated_ == 0.f));
  // Initial background mask
  background_mask_ = ~(accumulated_dZdt_ > tau_alpha);

  // Label objects
  cv::Mat object_label;
  auto object_num = cv::connectedComponents(~background_mask_,
                                            object_label,
                                            8,
                                            CV_16U);
  for (int object_idx = 1, update_derive = 1;
      object_idx < object_num; ++object_idx) {
    cv::Mat object_area = (object_label == object_idx);
    if (cv::sum(object_area)[0] < object_threshold_) {
      // Ignore small object segmentations
      accumulated_dZdt_.setTo(0, object_area);
      continue;
    }
    // Occlusion prediction, Equation 10
    accumInterpolation(object_area, new_discovered_area, update_derive);
    update_derive = 0;
  }

  // Equation 7
  object_mask = (accumulated_dZdt_ > tau_alpha);

  // Erase predicted areas that are not neighborhood of moving objects
  // Update predicted area
  predicted_area_ = (predicted_area_ + new_discovered_area).mul(object_mask);
  // Label predicted area
  cv::Mat& predicted_area_label = object_label;
  auto predicted_area_num = cv::connectedComponents(predicted_area_,
                                                    predicted_area_label,
                                                    8,
                                                    CV_16U);
  if (predicted_area_num > 1) {
    // Check predicted area is neighborhood of moving object
    auto connect_label = bwConnect(predicted_area_label,
                                   (object_mask - predicted_area_),
                                   predicted_area_num);
    // Erase unconnected area
    for (int label_idx = 1; label_idx < predicted_area_num; ++label_idx) {
      if (connect_label.at(label_idx) == 0) {
        object_mask.setTo(0, predicted_area_label == label_idx);
        accumulated_dZdt_.setTo(0, predicted_area_label == label_idx);
      }
    }
  }

  // Moving object detection result
  background_mask_ = ~object_mask;
}


void OcclusionAccumulation::DepthCompensation(const cv::Matx33f& R,
                                              const cv::Vec3f& t) {
  // Fill current depth by previous warped depth
  depth_prev_warped_.copyTo(depth_cur_compensated_,
                            depth_cur_compensated_ == 0.f);

  // Get current warped depth by current depth
  getWarpedImage(depth_cur_compensated_, depth_cur_compensated_,
                 depth_cur_compensated_, R.t(), -R.t()*t, depth_prev_warped_);

  // Fill next depth by current depth warped
  depth_prev_warped_.copyTo(depth_next_compensated_,
                            depth_next_compensated_ == 0.f);

  // Fill current warped depth by next depth
  depth_next_compensated_.copyTo(depth_prev_warped_,
                                 depth_prev_warped_ == 0.f);
}

void OcclusionAccumulation::getWarpedImage(cv::InputArray i_ref,
                                           cv::InputArray d_ref,
                                           cv::InputArray i,
                                           cv::InputArray R,
                                           cv::InputArray t,
                                           cv::InputOutputArray ir,
                                           cv::OutputArray residual,
                                           bool update_warped) {
  static cv::Mat x_coordinate, y_coordinate;

  cv::Mat i_ref_ = i_ref.getMat();
  cv::Mat d_ref_ = d_ref.getMat();
  cv::Mat i_ = i.getMat();
  cv::Matx33f R_ = R.getMat();
  cv::Vec3f t_ = t.getMat();
  if (ir.empty())
    ir.create(i_.size(), i_.type());
  cv::Mat ir_ = ir.getMatRef();
  cv::Mat residual_;
  if (residual.needed()) {
    residual.create(i_.size(), i_.type());
    residual_ = residual.getMatRef();
  }

  auto meshgrid = [](const cv::Range x, const cv::Range y,
                            cv::Mat& xmap, cv::Mat& ymap) {
    std::vector<float> index_x, index_y;
    index_x.reserve(x.size());
    index_y.reserve(y.size());
    for (auto index = x.start; index < x.end; ++index)
      index_x.push_back(float(index));
    for (auto index = y.start; index < y.end; ++index)
      index_y.push_back(float(index));

    cv::repeat(cv::Mat(index_x).t(), y.size(), 1, xmap);
    cv::repeat(cv::Mat(index_y), 1, x.size(), ymap);
  };

  auto getWarp = [&](cv::Mat& x, cv::Mat& y) {
    cv::Mat z;
    meshgrid(cv::Range(0, d_ref_.cols), cv::Range(0, d_ref_.rows), x, y);
    x = x.reshape(0, d_ref_.rows*d_ref_.cols);
    y = y.reshape(0, d_ref_.rows*d_ref_.cols);
    cv::Mat depth = d_ref_.reshape(0, d_ref_.rows*d_ref_.cols);

    x = ((x - K_(0, 2)) / K_(0, 0)).mul(depth);
    y = ((y - K_(1, 2)) / K_(1, 1)).mul(depth);

    x = R_(0, 0)*x + R_(0, 1)*y + R_(0, 2)*depth + t_[0];
    y = R_(1, 0)*x + R_(1, 1)*y + R_(1, 2)*depth + t_[1];
    z = R_(2, 0)*x + R_(2, 1)*y + R_(2, 2)*depth + t_[2];

    x = K_(0, 0) * x.mul(1.f / z) + K_(0, 2);
    y = K_(1, 1) * y.mul(1.f / z) + K_(1, 2);

    x = x.reshape(0, d_ref_.rows);
    y = y.reshape(0, d_ref_.rows);
  };

  if (update_warped)
    getWarp(x_coordinate, y_coordinate);
  cv::remap(i_, ir_, x_coordinate, y_coordinate, cv::INTER_NEAREST);
  if (residual.needed())
    residual_ = ir_ - i_ref_;

//  std::ofstream myfile;
//  myfile.open("dubug.csv");
//  myfile << cv::format(ir_, cv::Formatter::FMT_CSV) << std::endl;
//  myfile.close();
}

void OcclusionAccumulation::accumInterpolation(
    cv::InputOutputArray source_mask,
    cv::InputOutputArray target_mask,
    bool update_derive) {
  static cv::Mat dxn, dyn, dxp, dyp;
  auto& source_mask_ = source_mask.getMatRef();
  auto& target_mask_ = target_mask.getMatRef();

  auto getdxynp = [](cv::InputArray image,
                     cv::OutputArray dxn, cv::OutputArray dyn,
                     cv::OutputArray dxp, cv::OutputArray dyp) {
    if (dxn.empty())
      dxn.create(image.size(), image.type());
    if (dyn.empty())
      dyn.create(image.size(), image.type());
    if (dxp.empty())
      dxp.create(image.size(), image.type());
    if (dyp.empty())
      dyp.create(image.size(), image.type());

    auto image_ = image.getMat();
    auto dxn_ = dxn.getMatRef();
    auto dyn_ = dyn.getMatRef();
    auto dxp_ = dxp.getMatRef();
    auto dyp_ = dyp.getMatRef();

    const int rows = image_.rows, cols = image_.cols;
    dyp_.rowRange(0, rows - 1) =
        image_.rowRange(1, rows) - image_.rowRange(0, rows - 1);
    dxp_.colRange(0, cols - 1) =
        image_.colRange(1, cols) - image_.colRange(0, cols - 1);
    dyn_.rowRange(1, rows) =
        image_.rowRange(0, rows - 1) - image_.rowRange(1, rows);
    dxn_.colRange(1, cols) =
        image_.colRange(0, cols - 1) - image_.colRange(1, cols);
  };

  if (update_derive)
    getdxynp(depth_next_compensated_, dxn, dyn, dxp, dyp);
  float dxy_threshold = 0.05f;

  cv::Mat near_xp = cv::Mat::zeros(source_mask_.size(), CV_32F);
  cv::Mat near_yp = cv::Mat::zeros(source_mask_.size(), CV_32F);
  cv::Mat near_xn = cv::Mat::zeros(source_mask_.size(), CV_32F);
  cv::Mat near_yn = cv::Mat::zeros(source_mask_.size(), CV_32F);
  cv::Mat restoration = cv::Mat::zeros(source_mask_.size(), CV_32F);

  cv::Mat is_near_something = cv::Mat::zeros(source_mask_.size(), CV_32F);
  const int rows = source_mask_.rows, cols = source_mask_.cols;
  do {
    near_xp(cv::Range(1, rows-1), cv::Range(1, cols-1)) =
        (source_mask_(cv::Range(1, rows-1), cv::Range(2, cols)).mul(
            target_mask_(cv::Range(1, rows-1), cv::Range(1, cols-1)))).mul(
            cv::abs(dxp(cv::Range(1, rows-1), cv::Range(1, cols-1))) <
                dxy_threshold);
    near_yp(cv::Range(1, rows-1), cv::Range(1, cols-1)) =
        (source_mask_(cv::Range(2, rows), cv::Range(1, cols-1)).mul(
            target_mask_(cv::Range(1, rows-1), cv::Range(1, cols-1)))).mul(
            cv::abs(dyp(cv::Range(1, rows-1), cv::Range(1, cols-1))) <
                dxy_threshold);
    near_xn(cv::Range(1, rows-1), cv::Range(1, cols-1)) =
        (source_mask_(cv::Range(1, rows-1), cv::Range(0, cols-2)).mul(
            target_mask_(cv::Range(1, rows-1), cv::Range(1, cols-1)))).mul(
            cv::abs(dxn(cv::Range(1, rows-1), cv::Range(1, cols-1))) <
                dxy_threshold);
    near_yn(cv::Range(1, rows-1), cv::Range(1, cols-1)) =
        (source_mask_(cv::Range(0, rows-2), cv::Range(1, cols-1)).mul(
            target_mask_(cv::Range(1, rows-1), cv::Range(1, cols-1)))).mul(
            cv::abs(dyn(cv::Range(1, rows-1), cv::Range(1, cols-1))) <
                dxy_threshold);

    is_near_something = near_xp + near_yp + near_xn + near_yn;

    restoration(cv::Range(1, rows-1), cv::Range(1, cols-1)) =
        ( (accumulated_dZdt_(cv::Range(1, rows-1), cv::Range(2, cols)) +
            dxp(cv::Range(1, rows-1), cv::Range(1, cols-1))).mul(
                near_xp(cv::Range(1, rows-1), cv::Range(1, cols-1)))
        + (accumulated_dZdt_(cv::Range(2, rows), cv::Range(1, cols-1)) +
            dyp(cv::Range(1, rows-1), cv::Range(1, cols-1))).mul(
                near_yp(cv::Range(1, rows-1), cv::Range(1, cols-1)))
        + (accumulated_dZdt_(cv::Range(1, rows-1), cv::Range(0, cols-2)) +
            dxn(cv::Range(1, rows-1), cv::Range(1, cols-1))).mul(
                near_xn(cv::Range(1, rows-1), cv::Range(1, cols-1)))
        + (accumulated_dZdt_(cv::Range(0, rows-2), cv::Range(1, cols-1)) +
            dyn(cv::Range(1, rows-1), cv::Range(1, cols-1))).mul(
                near_yn(cv::Range(1, rows-1), cv::Range(1, cols-1)))
        ).mul(1./is_near_something(cv::Range(1, rows-1),cv::Range(1, cols-1)));

    cv::Mat mask = is_near_something != 0.f;
    restoration.copyTo(accumulated_dZdt_, mask);
    target_mask_.setTo(0, mask);
    source_mask_.setTo(1, mask);

  } while (cv::sum(is_near_something)[0] > 10);
}

std::vector<int> OcclusionAccumulation::bwConnect(cv::InputArray label,
                                                  cv::InputArray ref,
                                                  int number_of_label) {
  cv::Mat label_ = label.getMat();
  cv::Mat ref_ = ref.getMat();
  const int rows = label_.rows, cols = label_.cols;
  std::vector<int> result(number_of_label, 0);

  cv::Mat dx_l = cv::Mat::ones(label_.size(), CV_8U);
  cv::Mat dy_l = cv::Mat::ones(label_.size(), CV_8U);

  dx_l.colRange(1, cols-1) =
      (label_.colRange(2, cols) != label_.colRange(0, cols - 2));
  dy_l.rowRange(1, rows-1) =
      (label_.rowRange(2, rows) != label_.rowRange(0, rows - 2));
  cv::Mat dxy_l = (dx_l + dy_l) > 0;

  cv::Mat dx_r = cv::Mat::ones(label_.size(), CV_8U);
  cv::Mat dy_r = cv::Mat::ones(label_.size(), CV_8U);

  dx_r.colRange(1, cols-1) =
      (ref_.colRange(2, cols) != ref_.colRange(0, cols - 2));
  dy_r.rowRange(1, rows-1) =
      (ref_.rowRange(2, rows) != ref_.rowRange(0, rows - 2));
  cv::Mat dxy_r = (dx_r + dy_r) > 0;

  cv::Mat& dxy = dxy_l;
  dxy = dxy_l.mul(dxy_r);

  std::vector<cv::Point> idx;
  cv::findNonZero(dxy, idx);

  for (const auto& item : idx) {
    if (label_.at<uchar>(item))
      result.at(label_.at<uchar>(item)) = 1;
  }
  return result;
}
