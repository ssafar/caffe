#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class UnpoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnpoolingLayerTest()
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

  virtual ~UnpoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnpoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnpoolingLayerTest, TestSizeIsIncreased) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param =
    layer_param.mutable_unpooling_param();

  unpooling_param->set_kernel_size(2);

  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 12);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(UnpoolingLayerTest, TestDataIsCopiedForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param =
    layer_param.mutable_unpooling_param();

  unpooling_param->set_kernel_size(2);

  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // The same bottom data should be found at multiple places at the top.
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 0, 0),
      this->blob_bottom_->data_at(1, 2, 0, 0));
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 0, 0),
      this->blob_top_->data_at(1, 2, 1, 0));
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 0, 0),
      this->blob_top_->data_at(1, 2, 1, 1));

  // Same checks, this time not in the top left corner.
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 2, 1),
      this->blob_top_->data_at(1, 2, 4, 2));
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 2, 1),
      this->blob_top_->data_at(1, 2, 5, 2));
  EXPECT_EQ(this->blob_bottom_->data_at(1, 2, 2, 1),
      this->blob_top_->data_at(1, 2, 4, 3));
}

TYPED_TEST(UnpoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UnpoolingParameter* unpooling_param =
    layer_param.mutable_unpooling_param();

  unpooling_param->set_kernel_size(2);

  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
