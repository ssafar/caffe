#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Crop Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Crop Layer takes a single blob as output.";

  // Expand parameters to crop amounts.
  int crop_h_top = 0;
  int crop_h_bottom = 0;
  int crop_w_left = 0;
  int crop_w_right = 0;

  const CropParameter& crop_param = this->layer_param_.crop_param();
  CHECK_LE(crop_param.crop_h_size(), 2);

  if (crop_param.crop_h_size() == 1) {
    crop_h_top = crop_param.crop_h(0);
    crop_h_bottom = crop_param.crop_h(0);
  } else if (crop_param.crop_h_size() == 2) {
    crop_h_top = crop_param.crop_h(0);
    crop_h_bottom = crop_param.crop_h(1);
  }

  CHECK_LE(crop_param.crop_w_size(), 2);

  if (crop_param.crop_w_size() == 1) {
    crop_w_left = crop_param.crop_w(0);
    crop_w_right = crop_param.crop_w(0);
  } else if (crop_param.crop_h_size() == 2) {
    crop_w_left = crop_param.crop_w(0);
    crop_w_right = crop_param.crop_w(1);
  }
  // If parameters are empty, they all keep their default values.

  // Calculate crop limits.
  valid_h_begin = crop_h_top;
  valid_h_end = bottom[0]->height() - crop_h_bottom;
  CHECK_GT(valid_h_end - valid_h_begin, 0) << "Crop output height should be greater than zero";

  valid_w_begin = crop_w_left;
  valid_w_end = bottom[0]->width() - crop_w_right;
  CHECK_GT(valid_w_end - valid_w_begin, 0) << "Crop output width should be greater than zero";
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      valid_h_end - valid_h_begin,
      valid_w_end - valid_w_begin);
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = valid_h_begin; h < valid_h_end; ++h) {
        const Dtype* bottom_row_start = bottom_data + bottom[0]->offset(n, c, h, valid_w_begin);
        Dtype* top_row_start = top_data + top[0]->offset(n, c, h - valid_h_begin, 0);
        caffe_copy(valid_w_end - valid_w_begin, bottom_row_start, top_row_start);
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // TODO(ssafar): could be done more efficiently; does it count?
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = valid_h_begin; h < valid_h_end; ++h) {
        Dtype* bottom_row_start = bottom_diff + bottom[0]->offset(n, c, h, valid_w_begin);
        const Dtype* top_row_start = top_diff + top[0]->offset(n, c, h - valid_h_begin, 0);
        caffe_copy(valid_w_end - valid_w_begin, top_row_start, bottom_row_start);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(CropLayer);

}   // namespace caffe
