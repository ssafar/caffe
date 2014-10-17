#include <vector>

#include "caffe/vision_layers.hpp"
#include "caffe/layer.hpp"


namespace caffe {

template <typename Dtype>
void UnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // At the time being, we don't support switch info input, so the layer is one
  // to one.
  CHECK_EQ(bottom.size(), 1) << "Unpooling Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Unpooling Layer takes a single blob as output.";
  int kernel_size = this->layer_param_.unpooling_param().kernel_size();
  CHECK_GT(kernel_size, 0);
  kernel_h_ = kernel_size;
  kernel_w_ = kernel_size;
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height() * kernel_h_,
      bottom[0]->width() * kernel_h_);
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Iterate over the top, make slow steps on the bottom.
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {

      Dtype * const top_current_matrix = top_data + top[0]->offset(n, c, 0, 0);
      const Dtype* bottom_current_matrix = bottom_data + bottom[0]->offset(n, c, 0, 0);
      const int top_stride = top[0]->width();
      const int bottom_stride = bottom[0]->width();

      for (int h = 0; h < top[0]->height(); ++h) {
        for (int w = 0; w < top[0]->width(); ++w) {
          top_current_matrix[top_stride*h+w] = bottom_current_matrix[bottom_stride*(h / kernel_h_) + (w / kernel_w_)];
        }
      }
    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* const bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // Iterate over the top, add up values on the bottom.
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {

      const Dtype *top_current_matrix = top_diff + top[0]->offset(n, c, 0, 0);
      Dtype* const bottom_current_matrix = bottom_diff + bottom[0]->offset(n, c, 0, 0);
      const int top_stride = top[0]->width();
      const int bottom_stride = bottom[0]->width();

      for (int h = 0; h < top[0]->height(); ++h) {
        for (int w = 0; w < top[0]->width(); ++w) {
          bottom_current_matrix[bottom_stride*(h / kernel_h_) + (w / kernel_w_)] += top_current_matrix[top_stride*h+w];
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(UnpoolingLayer);
#endif

INSTANTIATE_CLASS(UnpoolingLayer);
REGISTER_LAYER_CLASS(UNPOOLING, UnpoolingLayer);
}  // namespace caffe
