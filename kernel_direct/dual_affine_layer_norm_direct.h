#ifndef DUAL_AFFINE_LAYER_NORM_DIRECT_H
#define DUAL_AFFINE_LAYER_NORM_DIRECT_H

#include <cstdint>

// dtype: 0=FP16, 1=BF16. All pointers must refer to the current device.
void LaunchDualAffineLayerNormDirect(
    int dtype, int64_t n, int64_t h, int64_t alignedH, int64_t rowsPerCore,
    int64_t blockDim, int64_t coresPerRow, int64_t sliceH, float epsilon,
    const void* x, const void* gamma1, const void* beta1,
    const void* gamma2, const void* beta2, void* y1, void* y2,
    void* stream);

#endif
