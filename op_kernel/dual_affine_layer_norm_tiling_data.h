#ifndef DUAL_AFFINE_LAYER_NORM_TILING_DATA_H
#define DUAL_AFFINE_LAYER_NORM_TILING_DATA_H

#include <stdint.h>

struct DualAffineLayerNormTilingData {
    int64_t n = 0;
    int64_t h = 0;
    int64_t alignedH = 64;
    int64_t rowsPerCore = 1;
    int64_t ubBytes = 0;
    int64_t blockDim = 1;
    int64_t coresPerRow = 1;
    int64_t sliceH = 0;
    float epsilon = 1e-5f;
};

#endif
