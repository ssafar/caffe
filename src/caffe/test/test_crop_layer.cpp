#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CropLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CropLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CropLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CropLayerTest, TestDtypesAndDevices);

TYPED_TEST(CropLayerTest, TestCroppedOutputSizesNoCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CropParameter* crop_param =
    layer_param.mutable_crop_param();

  crop_param->add_crop_h(0); // Explicit zero.
  // Width is set to zero implicitly.

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);

  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(CropLayerTest, TestSymmetricCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CropParameter* crop_param =
    layer_param.mutable_crop_param();

  crop_param->add_crop_h(2);
  crop_param->add_crop_w(1);

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // The original blob was 2x3x6x5, so the output height x width should be
  // (6-2*2)x(5-2*1)==2x3.
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(CropLayerTest, TestAsymmetricCrop) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CropParameter* crop_param =
    layer_param.mutable_crop_param();

  crop_param->add_crop_h(2);
  crop_param->add_crop_h(1);

  crop_param->add_crop_w(1);
  crop_param->add_crop_w(3);

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->height(), 3);  // 6 - 2 - 1
  EXPECT_EQ(this->blob_top_->width(), 1);  // 5 - 1 - 3
}

TYPED_TEST(CropLayerTest, TestValuesAreCopied) {
typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CropParameter* crop_param =
    layer_param.mutable_crop_param();

  crop_param->add_crop_h(2);
  crop_param->add_crop_h(1);

  crop_param->add_crop_w(1);
  crop_param->add_crop_w(3);

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Top left corner of the top layer.
  EXPECT_EQ(
    this->blob_bottom_->data_at(1, 2, 2, 1),
    this->blob_top_->data_at(1, 2, 0, 0));

  // Bottom right corner of the 6x5 matrix.
  EXPECT_EQ(
    this->blob_bottom_->data_at(1, 2, 4, 1),
    this->blob_top_->data_at(1, 2, 2, 0));
}

TYPED_TEST(CropLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  CropParameter* crop_param =
    layer_param.mutable_crop_param();

  crop_param->add_crop_h(2);
  crop_param->add_crop_w(1);

  CropLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
