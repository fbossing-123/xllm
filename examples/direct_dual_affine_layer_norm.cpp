#include "sample_common.h"
#include "dual_affine_layer_norm_direct.h"

#include <iostream>

int main(int argc, char** argv)
{
    try {
        if (argc != 7 && argc != 8) {
            std::cerr << "usage: direct_dual_affine_layer_norm <case_dir> <device> <float16|bfloat16> <N> <H> <epsilon> [core_num=48]\n";
            return 2;
        }
        const std::string directory = argv[1];
        const int device = std::stoi(argv[2]);
        const std::string dtypeName = argv[3];
        const aclDataType dtype = ParseDtype(dtypeName);
        const int64_t n = std::stoll(argv[4]);
        const int64_t h = std::stoll(argv[5]);
        const float epsilon = std::stof(argv[6]);
        const int64_t coreNum = argc == 8 ? std::stoll(argv[7]) : 48;
        if (n < 1 || n > 4096 || h < 1 || h > 8192 || coreNum < 1 ||
            !(epsilon >= 1e-7f && epsilon <= 1e-2f)) {
            throw std::invalid_argument("invalid N/H/core_num/epsilon");
        }
        const int64_t alignedH = (h + 63) / 64 * 64;
        int64_t blockDim = std::min(n, coreNum);
        const int64_t rowsPerCore = (n + blockDim - 1) / blockDim;
        blockDim = (n + rowsPerCore - 1) / rowsPerCore;

        SampleContext context(device);
        SampleTensor x(directory, "x", {n, h}, dtype, true);
        SampleTensor gamma1(directory, "gamma1", {h}, dtype, true);
        SampleTensor beta1(directory, "beta1", {h}, dtype, true);
        SampleTensor gamma2(directory, "gamma2", {h}, dtype, true);
        SampleTensor beta2(directory, "beta2", {h}, dtype, true);
        SampleTensor y1(directory, "y1", {n, h}, dtype, false);
        SampleTensor y2(directory, "y2", {n, h}, dtype, false);
        LaunchDualAffineLayerNormDirect(
            dtypeName == "bfloat16" ? 1 : 0, n, h, alignedH, rowsPerCore, blockDim, epsilon,
            x.data(), gamma1.data(), beta1.data(), gamma2.data(), beta2.data(), y1.data(), y2.data(),
            context.stream());
        CheckAcl(aclrtSynchronizeStream(context.stream()), "aclrtSynchronizeStream");
        y1.Save(directory, "y1");
        y2.Save(directory, "y2");
        std::cout << "Direct-kernel execution completed; workspace_size=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
