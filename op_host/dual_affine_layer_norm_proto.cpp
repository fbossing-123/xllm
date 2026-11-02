/*
 * Shape/dtype inference for DualAffineLayerNorm.
 */
#include "register/op_impl_registry.h"

namespace ops {

ge::graphStatus InferShapeDualAffineLayerNorm(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* y1Shape = context->GetOutputShape(0);
    gert::Shape* y2Shape = context->GetOutputShape(1);
    if (xShape == nullptr || y1Shape == nullptr || y2Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y1Shape = *xShape;
    *y2Shape = *xShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeDualAffineLayerNorm(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DualAffineLayerNorm)
    .InferShape(InferShapeDualAffineLayerNorm)
    .InferDataType(InferDataTypeDualAffineLayerNorm);

} // namespace ops
