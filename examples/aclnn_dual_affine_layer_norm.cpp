#include "sample_common.h"
#include "aclnn_dual_affine_layer_norm.h"

#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    try {
        if (argc != 7) {
            std::cerr << "usage: aclnn_dual_affine_layer_norm <case_dir> <device> <float16|bfloat16> <N> <H> <epsilon>\n";
            return 2;
        }
        const std::string directory = argv[1];
        const int device = std::stoi(argv[2]);
        const aclDataType dtype = ParseDtype(argv[3]);
        const int64_t n = std::stoll(argv[4]);
        const int64_t h = std::stoll(argv[5]);
        const double epsilon = std::stod(argv[6]);
        SampleContext context(device);
        SampleTensor x(directory, "x", {n, h}, dtype, true);
        SampleTensor gamma1(directory, "gamma1", {h}, dtype, true);
        SampleTensor beta1(directory, "beta1", {h}, dtype, true);
        SampleTensor gamma2(directory, "gamma2", {h}, dtype, true);
        SampleTensor beta2(directory, "beta2", {h}, dtype, true);
        SampleTensor y1(directory, "y1", {n, h}, dtype, false);
        SampleTensor y2(directory, "y2", {n, h}, dtype, false);

        uint64_t workspaceSize = 0;
        aclOpExecutor* rawExecutor = nullptr;
        CheckAcl(aclnnDualAffineLayerNormGetWorkspaceSize(
                     x.tensor(), gamma1.tensor(), beta1.tensor(), gamma2.tensor(), beta2.tensor(),
                     epsilon, y1.tensor(), y2.tensor(), &workspaceSize, &rawExecutor),
                 "aclnnDualAffineLayerNormGetWorkspaceSize");
        std::unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> executor(
            rawExecutor, aclDestroyAclOpExecutor);
        if (workspaceSize != 0) throw std::runtime_error("explicit workspace must be zero");
        CheckAcl(aclnnDualAffineLayerNorm(nullptr, workspaceSize, executor.get(), context.stream()),
                 "aclnnDualAffineLayerNorm");
        CheckAcl(aclrtSynchronizeStream(context.stream()), "aclrtSynchronizeStream");
        y1.Save(directory, "y1");
        y2.Save(directory, "y2");
        std::cout << "ACLNN execution completed; workspace_size=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
