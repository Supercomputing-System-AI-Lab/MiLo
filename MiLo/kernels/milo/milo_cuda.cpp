#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
int milo_cuda(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

int milo_cuda_with_zeros(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* z,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void mul_3bit(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = milo_cuda(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}


void mul_3bit_with_zeros(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
  const torch::Tensor& z,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = milo_cuda_with_zeros(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul_3bit", &mul_3bit, "MiLo FP16xINT3 matmul.");
  m.def("mul_3bit_with_zeros", &mul_3bit_with_zeros, "MiLo FP16xINT3 matmul with zeros.");
}
