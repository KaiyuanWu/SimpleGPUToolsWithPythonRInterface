#include "imageaugmenterlet.h"

ImageAugmenterLet::ImageAugmenterLet(ImageAugmentLetParam  param) {
  rotateM_ = cv::Mat(2, 3, CV_32F);
  param_ = param;
}


int ImageAugmenterLet::GetInterMethod(int inter_method, int old_width, int old_height, int new_width, int new_height){
      if (inter_method == 9) {
        if (new_width > old_width && new_height > old_height) {
          return 2;  // CV_INTER_CUBIC for enlarge
        } else if (new_width <old_width && new_height < old_height) {
          return 3;  // CV_INTER_AREA for shrink
        } else {
          return 1;  // CV_INTER_LINEAR for others
        }
        } else if (inter_method == 10) {
        std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
        return rand_uniform_int(generator);
      } else {
        return inter_method;
      }
}

cv::Mat ImageAugmenterLet::Process(const cv::Mat &src, int nlandmarks, float *landmarks_x, float *landmarks_y)  {
  cv::Mat res;
  // normal augmentation by affine transformation.
  if (param_.max_rotate_angle > 0 || param_.max_shear_ratio > 0.0f
      || param_.rotate > 0 || rotate_list_.size() > 0 || param_.max_random_scale != 1.0
      || param_.min_random_scale != 1.0 || param_.max_aspect_ratio != 0.0f
      || param_.max_img_size != 1e10f || param_.min_img_size != 0.0f) {
    std::uniform_real_distribution<float> rand_uniform(0, 1);
    // shear
    float s = rand_uniform(generator) * param_.max_shear_ratio * 2 - param_.max_shear_ratio;
    // rotate
    int angle = std::uniform_int_distribution<int>(
        -param_.max_rotate_angle, param_.max_rotate_angle)(generator);
    if (param_.rotate > 0) angle = param_.rotate;
    if (rotate_list_.size() > 0) {
      angle = rotate_list_[std::uniform_int_distribution<int>(0, rotate_list_.size() - 1)(generator)];
    }
    float a = cos(angle / 180.0 * M_PI);
    float b = sin(angle / 180.0 * M_PI);
    // scale
    float scale = rand_uniform(generator) *
        (param_.max_random_scale - param_.min_random_scale) + param_.min_random_scale;
    // aspect ratio
    float ratio = rand_uniform(generator) *
        param_.max_aspect_ratio * 2 - param_.max_aspect_ratio + 1;
    float hs = 2 * scale / (1 + ratio);
    float ws = ratio * hs;
    // new width and height
    float new_width = std::max(param_.min_img_size,
                               std::min(param_.max_img_size, scale * src.cols));
    float new_height = std::max(param_.min_img_size,
                                std::min(param_.max_img_size, scale * src.rows));
    rotateM_.at<float>(0, 0) = hs * a - s * b * ws;
    rotateM_.at<float>(1, 0) = -b * ws;
    rotateM_.at<float>(0, 1) = hs * b + s * a * ws;
    rotateM_.at<float>(1, 1) = a * ws;
    float ori_center_width = rotateM_.at<float>(0, 0) * src.cols + rotateM_.at<float>(0, 1) * src.rows;
    float ori_center_height =rotateM_.at<float>(1, 0) * src.cols + rotateM_.at<float>(1, 1) * src.rows;
    rotateM_.at<float>(0, 2) = (new_width - ori_center_width) / 2;
    rotateM_.at<float>(1, 2) = (new_height - ori_center_height) / 2;
    if(!((param_.inter_method >= 1 && param_.inter_method <= 4) ||
       (param_.inter_method >= 9 && param_.inter_method <= 10))){
       std::cout << "invalid inter_method: valid value 0,1,2,3,9,10"<<std::endl;
       exit(-1);
    }
    int interpolation_method = GetInterMethod(param_.inter_method,
                   src.cols, src.rows, new_width, new_height);
    cv::warpAffine(src, temp_, rotateM_, cv::Size(new_width, new_height),
                   interpolation_method,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
    res = temp_;
  } else {
      rotateM_.at<float>(0, 0) = 0;
      rotateM_.at<float>(1, 0) = 0;
      rotateM_.at<float>(0, 1) = 0;
      rotateM_.at<float>(1, 1) = 0;
     rotateM_.at<float>(0, 2) = 0;
     rotateM_.at<float>(1, 2) = 0;
    res = src;
  }


  // pad logic
  if (param_.pad > 0) {
    cv::copyMakeBorder(res, res, param_.pad, param_.pad, param_.pad, param_.pad,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(param_.fill_value, param_.fill_value, param_.fill_value));
     rotateM_.at<float>(0, 2) += param_.pad;
     rotateM_.at<float>(1, 2) += param_.pad;
  }

  // crop logic
  if (param_.max_crop_size != -1 || param_.min_crop_size != -1) {
    if(!(res.cols >= param_.max_crop_size && res.rows >= \
            param_.max_crop_size && param_.max_crop_size >= param_.min_crop_size)){
       std::cout << "input image size smaller than max_crop_size"<<std::endl;
       exit(-1);
    }
    int rand_crop_size =  std::uniform_int_distribution<int>(param_.min_crop_size, param_.max_crop_size)(generator);
    std::cout<<"rand_crop_size = "<<rand_crop_size<<std::endl;
    int y = res.rows - rand_crop_size;
    int x = res.cols - rand_crop_size;
    if (param_.rand_crop != 0) {
      y = std::uniform_int_distribution<int>(0, y)(generator);
      x = std::uniform_int_distribution<int>(0, x)(generator);
    } else {
      y /= 2; x /= 2;
    }
    cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
    int interpolation_method = GetInterMethod(param_.inter_method, rand_crop_size, rand_crop_size,
                                              param_.data_shape[2], param_.data_shape[1]);
    cv::resize(res(roi), res, cv::Size(param_.data_shape[2], param_.data_shape[1])
              , 0, 0, interpolation_method);
    rotateM_.at<float>(0, 2) -= x;
    rotateM_.at<float>(1, 2) -= y;

    rotateM_.at<float>(0, 0) *= float(param_.data_shape[2])/rand_crop_size;
    rotateM_.at<float>(0, 1) *= float(param_.data_shape[2])/rand_crop_size;
    rotateM_.at<float>(0, 2) *= float(param_.data_shape[2])/rand_crop_size;

    rotateM_.at<float>(1, 0) *= float(param_.data_shape[1])/rand_crop_size;
    rotateM_.at<float>(1, 1) *= float(param_.data_shape[1])/rand_crop_size;
    rotateM_.at<float>(1, 2) *= float(param_.data_shape[1])/rand_crop_size;
  } else {
    if(!(static_cast<int>(res.rows) >= param_.data_shape[1]
          && static_cast<int>(res.cols) >= param_.data_shape[2])){
       std::cout << "input image size smaller than input shape"<<std::endl;
       exit(-1);
    }
    int y = res.rows - param_.data_shape[1];
    int x = res.cols - param_.data_shape[2];
    if (param_.rand_crop != 0) {
      y = std::uniform_int_distribution<int>(0, y)(generator);
      x = std::uniform_int_distribution<int>(0, x)(generator);
    } else {
      y /= 2; x /= 2;
    }
    cv::Rect roi(x, y, param_.data_shape[2], param_.data_shape[1]);
    res = res(roi);
    rotateM_.at<float>(0, 2) -= x;
    rotateM_.at<float>(1, 2) -= y;
  }

  // color space augmentation
  if (param_.random_h != 0 || param_.random_s != 0 || param_.random_l != 0) {
    std::uniform_real_distribution<float> rand_uniform(0, 1);
    cv::cvtColor(res, res, CV_BGR2HLS);
    int h = rand_uniform(generator) * param_.random_h * 2 - param_.random_h;
    int s = rand_uniform(generator) * param_.random_s * 2 - param_.random_s;
    int l = rand_uniform(generator) * param_.random_l * 2 - param_.random_l;
    int temp[3] = {h, l, s};
    int limit[3] = {180, 255, 255};
    for (int i = 0; i < res.rows; ++i) {
      for (int j = 0; j < res.cols; ++j) {
        for (int k = 0; k < 3; ++k) {
          int v = res.at<cv::Vec3b>(i, j)[k];
          v += temp[k];
          v = std::max(0, std::min(limit[k], v));
          res.at<cv::Vec3b>(i, j)[k] = v;
        }
      }
    }
    cv::cvtColor(res, res, CV_HLS2BGR);
  }
  if(param_.rand_mirror){
        if(std::uniform_int_distribution<int>(0, 1)(generator) == 1){
            rotateM_.at<float>(0, 0) *= -1.f;
            rotateM_.at<float>(0, 1) *= -1.f;
            rotateM_.at<float>(0, 2) =  res.cols - rotateM_.at<float>(0, 2);
            cv::flip(res, res, 1);
        }
  }

  if(nlandmarks > 0){
      float tx, ty;
      float a11 = rotateM_.at<float>(0, 0);
      float a12 = rotateM_.at<float>(0, 1);
      float b1 = rotateM_.at<float>(0, 2);
      float a21 = rotateM_.at<float>(1, 0);
      float a22 = rotateM_.at<float>(1, 1);
      float b2 = rotateM_.at<float>(1, 2);
      for(int ilandmark = 0;  ilandmark < nlandmarks; ilandmark++){
         tx = a11* landmarks_x[ilandmark] + a12* landmarks_y[ilandmark] + b1;
         ty = a21* landmarks_x[ilandmark] + a22* landmarks_y[ilandmark] + b2;
         landmarks_x[ilandmark]  = tx;
         landmarks_y[ilandmark] = ty;
      }
  }
  return res;
}
