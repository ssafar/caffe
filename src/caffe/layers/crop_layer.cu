#include "caffe/layer.hpp"

#include "caffe/vision_layers.hpp"

namespace caffe {
/**
 * GPU implementation of the forward pass of image cropping.
 *
 * @param nthreads the number of threads overall (should be equal to top[0]->count()
 * @param bottom_data pointer to the GPU data of the bottom layer
 * TODO(ssafar): to be done.
 */
template <typename Dtype>
__global__ void CropForward(const int nthreads,
    const Dtype* bottom_data,
    const int bottom_n, const int bottom_c, const int bottom_h,
    const int bottom_w,
    Dtype* const top_data,
    const int valid_h_begin, const int top_h,
    const int valid_w_begin, const int top_w) {

  const int top_c = bottom_c;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int top_pos_w = index % top_w;
    const int top_pos_h = index / top_w % top_h;
    const int pos_c = index / top_w / top_h % top_c;
    const int pos_n = index / top_w / top_h / top_c;


    const int bottom_pos_w = top_pos_w + valid_w_begin;
    const int bottom_pos_h = top_pos_h + valid_h_begin;

    Dtype* const top_addr = top_data + index;

    int bottom_offset = pos_n;
    bottom_offset = bottom_offset * bottom_c + pos_c;
    bottom_offset = bottom_offset * bottom_h + bottom_pos_h;
    bottom_offset = bottom_offset * bottom_w + bottom_pos_w;
    const Dtype* bottom_addr = bottom_data + bottom_offset;

    *top_addr = *bottom_addr;
  }
}

template <typename Dtype>
__global__ void CropBackward(const int nthreads,
    Dtype* const bottom_diff,
    const int bottom_n, const int bottom_c, const int bottom_h,
    const int bottom_w,
    const Dtype* top_diff,
    const int valid_h_begin, const int top_h,
    const int valid_w_begin, const int top_w) {
  const int top_c = bottom_c;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int bottom_pos_w = index % bottom_w;
    const int bottom_pos_h = index / bottom_w % bottom_h;
    const int pos_c = index / bottom_w / bottom_h % bottom_c;
    const int pos_n = index / bottom_w / bottom_h / bottom_c;

    const int top_pos_h = bottom_pos_h - valid_h_begin;
    const int top_pos_w = bottom_pos_w - valid_w_begin;

    Dtype* const bottom_addr = bottom_diff + index;

    if (top_pos_w >= 0 && top_pos_w < top_w &&
        top_pos_h >= 0 && top_pos_h < top_h) {

      int top_offset = pos_n;
      top_offset = top_offset * top_c + pos_c;
      top_offset = top_offset * top_h + top_pos_h;
      top_offset = top_offset * top_w + top_pos_w;
      const Dtype* top_addr = top_diff + top_offset;

      *bottom_addr = *top_addr;
    } else {
      *bottom_addr = 0;
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  // One thread for each output (top) value.
  int count = top[0]->count();

  CropForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data,
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    top_data,
    valid_h_begin, top[0]->height(), valid_w_begin, top[0]->width());
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* const bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  // One thread for each output (bottom) value.
  int count = bottom[0]->count();

  CropBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_diff,
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
    top_diff,
    valid_h_begin, top[0]->height(), valid_w_begin, top[0]->width());
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
