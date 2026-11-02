#include "dual_affine_layer_norm_direct.h"

#include <stdexcept>

#include "dual_affine_layer_norm.h"

using NsDualAffineLayerNorm::DualAffineLayerNorm;

template <typename T>
__aicore__ inline void RunDualAffineLayerNormDirect(
    GM_ADDR x, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
    GM_ADDR y1, GM_ADDR y2, int64_t n, int64_t h, int64_t alignedH,
    int64_t rowsPerCore, int64_t coresPerRow, int64_t sliceH, float epsilon)
{
    DualAffineLayerNormTilingData tilingData;
    tilingData.n = n;
    tilingData.h = h;
    tilingData.alignedH = alignedH;
    tilingData.rowsPerCore = rowsPerCore;
    tilingData.ubBytes = 14 * alignedH + 8 * ((sliceH + 63) / 64 * 64);
    tilingData.coresPerRow = coresPerRow < 1 ? 1 : coresPerRow;
    tilingData.sliceH = sliceH < 1 ? h : sliceH;
    tilingData.epsilon = epsilon;
    DualAffineLayerNorm<T> op;
    op.Init(x, gamma1, beta1, gamma2, beta2, y1, y2, &tilingData);
    op.Process();
}

__global__ __aicore__ void dual_affine_layer_norm_direct_fp16(
    GM_ADDR x, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
    GM_ADDR y1, GM_ADDR y2, int64_t n, int64_t h, int64_t alignedH,
    int64_t rowsPerCore, int64_t coresPerRow, int64_t sliceH, float epsilon)
{
    RunDualAffineLayerNormDirect<half>(x, gamma1, beta1, gamma2, beta2,
        y1, y2, n, h, alignedH, rowsPerCore, coresPerRow, sliceH, epsilon);
}

__global__ __aicore__ void dual_affine_layer_norm_direct_bf16(
    GM_ADDR x, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
    GM_ADDR y1, GM_ADDR y2, int64_t n, int64_t h, int64_t alignedH,
    int64_t rowsPerCore, int64_t coresPerRow, int64_t sliceH, float epsilon)
{
    RunDualAffineLayerNormDirect<bfloat16_t>(x, gamma1, beta1, gamma2, beta2,
        y1, y2, n, h, alignedH, rowsPerCore, coresPerRow, sliceH, epsilon);
}

void LaunchDualAffineLayerNormDirect(
    int dtype, int64_t n, int64_t h, int64_t alignedH, int64_t rowsPerCore,
    int64_t blockDim, int64_t coresPerRow, int64_t sliceH, float epsilon,
    const void* x, const void* gamma1, const void* beta1,
    const void* gamma2, const void* beta2, void* y1, void* y2,
    void* stream)
{
    if (blockDim < 1 || blockDim > 4096) {
        throw std::invalid_argument("invalid direct blockDim");
    }
    if (dtype == 0) {
        dual_affine_layer_norm_direct_fp16<<<static_cast<uint32_t>(blockDim), nullptr, stream>>>(
            (GM_ADDR)const_cast<void*>(x), (GM_ADDR)const_cast<void*>(gamma1),
            (GM_ADDR)const_cast<void*>(beta1), (GM_ADDR)const_cast<void*>(gamma2),
            (GM_ADDR)const_cast<void*>(beta2), (GM_ADDR)y1, (GM_ADDR)y2,
            n, h, alignedH, rowsPerCore, coresPerRow, sliceH, epsilon);
    } else if (dtype == 1) {
        dual_affine_layer_norm_direct_bf16<<<static_cast<uint32_t>(blockDim), nullptr, stream>>>(
            (GM_ADDR)const_cast<void*>(x), (GM_ADDR)const_cast<void*>(gamma1),
            (GM_ADDR)const_cast<void*>(beta1), (GM_ADDR)const_cast<void*>(gamma2),
            (GM_ADDR)const_cast<void*>(beta2), (GM_ADDR)y1, (GM_ADDR)y2,
            n, h, alignedH, rowsPerCore, coresPerRow, sliceH, epsilon);
    } else {
        throw std::invalid_argument("direct dtype must be FP16 or BF16");
    }
}
