#ifndef ImageAugmenterLet_H
#define ImageAugmenterLet_H
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

struct ImageAugmentLetParam {
    ImageAugmentLetParam (){
        rand_crop = false;
        crop_y_start = -1;
        crop_x_start = -1;
        max_rotate_angle = 0.f;
        max_aspect_ratio = 0.f;
        max_shear_ratio = 0.f;
        max_crop_size = -1;
        min_crop_size = -1;
        max_random_scale = 1;
        min_random_scale = 1.;
        max_img_size = 1e10f;
        min_img_size = 0.0f;
        random_h = 0.f;
        random_s = 0.f;
        random_l = 0.f;
        rotate = -1.f;
        fill_value = 255;
        inter_method = 1;
        pad = 0;
        rand_mirror = false;
    }
  bool rand_mirror;
  /*! \brief whether we do random cropping */
  bool rand_crop;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle;
  /*! \brief max aspect ratio */
  float max_aspect_ratio;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio;
  /*! \brief max crop size */
  int max_crop_size;
  /*! \brief min crop size */
  int min_crop_size;
  /*! \brief max scale ratio */
  float max_random_scale;
  /*! \brief min scale_ratio */
  float min_random_scale;
  /*! \brief min image size */
  float min_img_size;
  /*! \brief max image size */
  float max_img_size;
  /*! \brief max random in H channel */
  int random_h;
  /*! \brief max random in S channel */
  int random_s;
  /*! \brief max random in L channel */
  int random_l;
  /*! \brief rotate angle */
  int rotate;
  /*! \brief filled color while padding */
  int fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief padding size */
  int pad;
  /*! \brief shape of the image data*/
  int data_shape[3];
};

class ImageAugmenterLet
{
public:
    ImageAugmenterLet(ImageAugmentLetParam  param);
    cv::Mat Process(const cv::Mat &src, int nlandmarks = 0 , float* landmarks_x = NULL, float*landmarks_y = NULL);
private:
    /*!
     * \brief get interpolation method with given inter_method, 0-CV_INTER_NN 1-CV_INTER_LINEAR 2-CV_INTER_CUBIC
     * \ 3-CV_INTER_AREA 4-CV_INTER_LANCZOS4 9-AUTO(cubic for enlarge, area for shrink, bilinear for others) 10-RAND
     */
    int GetInterMethod(int inter_method,  int old_width, int old_height, int new_width, int new_height);
    std::default_random_engine generator;
    // temporal space
    cv::Mat temp_;
    // rotation param
    cv::Mat rotateM_;
    // parameters
    ImageAugmentLetParam  param_;
    /*! \brief list of possible rotate angle */
    std::vector<int> rotate_list_;
};

#endif // ImageAugmenterLet_H
