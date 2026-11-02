#include <algorithm>

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"
#include "../op_kernel/dual_affine_layer_norm_tiling_data.h"
#include "../op_kernel/dual_affine_layer_norm_tiling_key.h"

namespace optiling {

ge::graphStatus DualAffineLayerNormTiling(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = inputShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "x must be rank 2"), return ge::GRAPH_FAILED);
    const int64_t n = shape.GetDim(0);
    const int64_t h = shape.GetDim(1);
    OP_CHECK_IF(n < 1 || n > 4096 || h < 1 || h > 8192,
                OP_LOGE(context, "N/H outside supported range"), return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum < 1 || ubSize == 0, OP_LOGE(context, "invalid platform"), return ge::GRAPH_FAILED);

    const int64_t alignedH = (h + 63) / 64 * 64;
    constexpr int64_t kMinSlice = 512;
    int64_t coresPerRow = 1;
    if (n < coreNum) {
        int64_t wanted = h / kMinSlice;
        if (wanted < 1) {
            wanted = 1;
        }
        coresPerRow = std::min(coreNum / n, wanted);
        if (coresPerRow < 1) {
            coresPerRow = 1;
        }
    }
    int64_t sliceH = h;
    int64_t rowsPerCore = 1;
    int64_t blockDim = 1;
    if (coresPerRow > 1) {
        sliceH = (h + coresPerRow - 1) / coresPerRow;
        sliceH = (sliceH + 63) / 64 * 64;
        coresPerRow = (h + sliceH - 1) / sliceH;
        blockDim = n * coresPerRow;
    } else {
        coresPerRow = 1;
        sliceH = h;
        blockDim = std::min(n, coreNum);
        rowsPerCore = (n + blockDim - 1) / blockDim;
        blockDim = (n + rowsPerCore - 1) / rowsPerCore;
    }
    const int64_t sliceAligned = (sliceH + 63) / 64 * 64;
    const int64_t ubBytes = 14 * alignedH + 8 * sliceAligned;
    OP_CHECK_IF(ubBytes > static_cast<int64_t>(ubSize),
                OP_LOGE(context, "UB budget exceeded: %ld > %lu", ubBytes, ubSize), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockDim < 1 || blockDim > coreNum,
                OP_LOGE(context, "invalid blockDim: %ld (coreNum=%ld)", blockDim, coreNum), return ge::GRAPH_FAILED);

    DualAffineLayerNormTilingData* tiling = context->GetTilingData<DualAffineLayerNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->n = n;
    tiling->h = h;
    tiling->alignedH = alignedH;
    tiling->rowsPerCore = rowsPerCore;
    tiling->ubBytes = ubBytes;
    tiling->blockDim = blockDim;
    tiling->coresPerRow = coresPerRow;
    tiling->sliceH = sliceH;
    const auto attrs = context->GetAttrs();
    const float* epsilon = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);
    tiling->epsilon = epsilon == nullptr ? 1e-5f : *epsilon;
    OP_CHECK_IF(!(tiling->epsilon >= 1e-7f && tiling->epsilon <= 1e-2f),
                OP_LOGE(context, "epsilon outside supported range"), return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(DUAL_AFFINE_LAYER_NORM_FP16));
    } else if (dtype == ge::DT_BF16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(DUAL_AFFINE_LAYER_NORM_BF16));
    } else {
        OP_LOGE(context, "only FP16/BF16 are supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseDualAffineLayerNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DualAffineLayerNormCompileInfo {};

IMPL_OP_OPTILING(DualAffineLayerNorm)
    .Tiling(DualAffineLayerNormTiling)
    .TilingParse<DualAffineLayerNormCompileInfo>(TilingParseDualAffineLayerNorm);

} // namespace optiling
