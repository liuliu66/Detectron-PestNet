#include "caffe2/core/context_gpu.h"
#include "mean_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Mean, MeanOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(MeanGradient, MeanGradientOp<CUDAContext>);

} // namespace caffe2