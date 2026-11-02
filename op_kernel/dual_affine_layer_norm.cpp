#include "dual_affine_layer_norm.h"
#include "dual_affine_layer_norm_tiling_key.h"

using NsDualAffineLayerNorm::DualAffineLayerNorm;

template <uint32_t schMode>
__global__ __aicore__ void dual_affine_layer_norm(
    GM_ADDR x, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
    GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DualAffineLayerNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(DualAffineLayerNormTilingData, tilingData, tiling);
    if constexpr (schMode == DUAL_AFFINE_LAYER_NORM_FP16) {
        DualAffineLayerNorm<half> op;
        op.Init(x, gamma1, beta1, gamma2, beta2, y1, y2, &tilingData);
        op.Process();
    } else {
        DualAffineLayerNorm<bfloat16_t> op;
        op.Init(x, gamma1, beta1, gamma2, beta2, y1, y2, &tilingData);
        op.Process();
    }
}
