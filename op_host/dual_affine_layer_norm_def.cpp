#include "register/op_def_registry.h"

namespace ops {

class DualAffineLayerNorm : public OpDef {
public:
    explicit DualAffineLayerNorm(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> dtypes{ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::Format> formats{ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("x").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("gamma1").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta1").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("gamma2").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta2").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Output("y1").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Output("y2").ParamType(REQUIRED).DataType(dtypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DualAffineLayerNorm);

} // namespace ops
