#include "caffe/layer.hpp"

#include "caffe/vision_layers.hpp"

namespace caffe {

inline __device__ int offset_4d(int n, int c, int h, int w,
    int size_c, int size_h, int size_w) {
  return ((((n * size_c) + c) * size_h) + h) * size_w + w;
}

template <typename Dtype>
__global__ void UnpoolForward(const int nthreads,
    const Dtype* const bottom_data,
    const int bottom_n, const int bottom_c, const int bottom_h,
    const int bottom_w,
    Dtype* const top_data,
    const int kernel_w, const int kernel_h)

{
  const int top_c = bottom_c;
  const int top_w = bottom_w * kernel_w;
  const int top_h = bottom_h * kernel_h;

  CUDA_KERNEL_LOOP(index, nthreads) {
    // Go over the top layer, do the same thing as we did before.
    const int top_pos_w = index % top_w;
    const int top_pos_h = index / top_w % top_h;
    const int pos_c = index / top_w / top_h % top_c;
    const int pos_n = index / top_w / top_h / top_c;

    const int bottom_pos_w = top_pos_w / kernel_w;
    const int bottom_pos_h = top_pos_h / kernel_h;

    Dtype* const top_addr = top_data + index;

    // int bottom_offset = pos_n;
    // bottom_offset = bottom_offset * bottom_c + pos_c;
    // bottom_offset = bottom_offset * bottom_h + bottom_pos_h;
    // bottom_offset = bottom_offset * bottom_w + bottom_pos_w;
    const int bottom_offset = offset_4d(pos_n, pos_c, bottom_pos_h, bottom_pos_w,
        bottom_c, bottom_h, bottom_w);

    const Dtype* bottom_addr = bottom_data + bottom_offset;

    *top_addr = *bottom_addr;
  }
}

template <typename Dtype>
__global__ void UnpoolBackward(const int nthreads,
    Dtype* const bottom_diff,
    const int bottom_n, const int bottom_c, const int bottom_h,
    const int bottom_w,
    const Dtype* top_diff,
    const int kernel_w, const int kernel_h) {
  // Go along the bottom, collecting individual components using a loop. Not
  // super efficient, but at least works.
  const int top_c = bottom_c;
  const int top_h = bottom_h * kernel_h;
  const int top_w = bottom_w * kernel_w;

  CUDA_KERNEL_LOOP(index, nthreads) {
    // Go over the top layer, do the same thing as we did before.
    const int bottom_pos_w = index % bottom_w;
    const int bottom_pos_h = index / bottom_w % bottom_h;
    const int pos_c = index / bottom_w / bottom_h % bottom_c;
    const int pos_n = index / bottom_w / bottom_h / bottom_c;

    const int top_base_pos_h = bottom_pos_h * kernel_h;
    const int top_base_pos_w = bottom_pos_w * kernel_w;

    const Dtype* const top_base_addr = top_diff + offset_4d(pos_n, pos_c,
        top_base_pos_h, top_base_pos_w,
        top_c, top_h, top_w);

    Dtype* const bottom_addr = bottom_diff + offset_4d(pos_n, pos_c,
        bottom_pos_h, bottom_pos_w,
        bottom_c, bottom_h, bottom_w);

    *bottom_addr = 0;

    for (int delta_h = 0; delta_h < kernel_h; ++delta_h) {
      for (int delta_w = 0; delta_w < kernel_w; ++delta_w) {
        const Dtype* const top_addr = top_base_addr + top_w * delta_h + delta_w;
        *bottom_addr += *top_addr;
      }
    }
  }
}



template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // One thread for each output (top) value.
  int count = top[0]->count();

  UnpoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data,
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    top_data,
    kernel_h_, kernel_w_);

}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  // One thread for each output (bottom) value.
  int count = bottom[0]->count();

  UnpoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_diff,
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    top_diff,
    kernel_h_, kernel_w_);
}

INSTANTIATE_LAYER_GPU_FUNCS(UnpoolingLayer);

}  // namespace caffe

