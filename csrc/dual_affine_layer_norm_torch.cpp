#include <torch/extension.h>

#include <acl/acl.h>
#include <aclnn_dual_affine_layer_norm.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <c10/core/DeviceGuard.h>

#include <ATen/EmptyTensor.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "dual_affine_layer_norm_direct.h"

namespace {

constexpr int64_t kVectorCoreCount = 48;
constexpr int64_t kUbSize = 196608;
constexpr const char* kPlanCapsuleName = "dual_affine_layer_norm_plan";
constexpr size_t kMaxAclRank = 2;
constexpr size_t kTensorCount = 7;

struct Plan {
    int64_t n;
    int64_t h;
    int64_t alignedH;
    int64_t rowsPerCore;
    int64_t blockDim;
    int64_t ubBytes;
    int64_t coresPerRow;
    int64_t sliceH;
    double epsilon;
    at::ScalarType dtype;
    c10::DeviceIndex device;
};

Plan MakePlan(const std::vector<at::Tensor>& inputs, double epsilon)
{
    TORCH_CHECK(inputs.size() == 5, "inputs must contain exactly five tensors");
    const auto& x = inputs[0];
    TORCH_CHECK(x.defined(), "inputs must be defined tensors");
    TORCH_CHECK(x.dim() == 2, "x must be a rank-2 [N,H] tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "all tensors must have float16 or bfloat16 dtype");
    for (const auto& input : inputs) {
        TORCH_CHECK(input.defined(), "inputs must be defined tensors");
        TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                    "all tensors must be NPU tensors using PrivateUse1");
        TORCH_CHECK(input.device() == x.device(), "all tensors must be on the same device");
        TORCH_CHECK(input.scalar_type() == x.scalar_type(), "all tensors must have the same dtype");
        TORCH_CHECK(input.layout() == at::kStrided && input.is_contiguous(),
                    "all tensors must be dense ND contiguous");
    }
    const int64_t n = x.size(0);
    const int64_t h = x.size(1);
    TORCH_CHECK(n >= 1 && n <= 4096, "N must be in [1,4096]");
    TORCH_CHECK(h >= 1 && h <= 8192, "H must be in [1,8192]");
    for (size_t i = 1; i < inputs.size(); ++i) {
        TORCH_CHECK(inputs[i].dim() == 1 && inputs[i].size(0) == h,
                    "affine parameters must be rank-1 tensors of length H");
    }
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 1e-7 && epsilon <= 1e-2,
                "epsilon must be finite and in [1e-7,1e-2]");
    const int64_t alignedH = (h + 63) / 64 * 64;
    constexpr int64_t kMinSlice = 512;
    int64_t coresPerRow = 1;
    if (n < kVectorCoreCount) {
        int64_t wanted = h / kMinSlice;
        if (wanted < 1) {
            wanted = 1;
        }
        coresPerRow = std::min(kVectorCoreCount / n, wanted);
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
        blockDim = std::min(n, kVectorCoreCount);
        rowsPerCore = (n + blockDim - 1) / blockDim;
        blockDim = (n + rowsPerCore - 1) / rowsPerCore;
    }
    const int64_t sliceAligned = (sliceH + 63) / 64 * 64;
    const int64_t ubBytes = 14 * alignedH + 8 * sliceAligned;
    TORCH_CHECK(ubBytes <= kUbSize, "row does not fit the UB budget");
    return {n, h, alignedH, rowsPerCore, blockDim, ubBytes, coresPerRow, sliceH,
            epsilon, x.scalar_type(), x.device().index()};
}

void CheckAgainstPlan(const Plan& actual, const Plan& expected)
{
    TORCH_CHECK(actual.n == expected.n && actual.h == expected.h &&
                actual.dtype == expected.dtype && actual.device == expected.device &&
                actual.epsilon == expected.epsilon,
                "prepared plan requires the original N, H, dtype, device, and epsilon");
}

// Rank-2 descriptors fit on the stack: two heap vectors per descriptor were a
// measurable part of the per-call host cost.
aclTensor* CreateAclTensor(const at::Tensor& tensor)
{
    const int64_t rank = tensor.dim();
    TORCH_CHECK(rank >= 0 && rank <= static_cast<int64_t>(kMaxAclRank),
                "acl descriptor rank must not exceed two");
    int64_t shape[kMaxAclRank] = {1, 1};
    int64_t strides[kMaxAclRank] = {1, 1};
    for (int64_t i = 0; i < rank; ++i) {
        shape[i] = tensor.size(i);
    }
    for (int64_t i = rank; i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    const aclDataType dtype = tensor.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    return aclCreateTensor(shape, static_cast<uint64_t>(rank), dtype, strides, 0,
                           ACL_FORMAT_ND, shape, static_cast<uint64_t>(rank), tensor.data_ptr());
}

class AclTensorSet {
public:
    AclTensorSet() = default;
    ~AclTensorSet() { Destroy(); }
    AclTensorSet(const AclTensorSet&) = delete;
    AclTensorSet& operator=(const AclTensorSet&) = delete;

    void Add(const at::Tensor& tensor)
    {
        TORCH_CHECK(size_ < kTensorCount, "too many acl tensors");
        aclTensor* value = CreateAclTensor(tensor);
        TORCH_CHECK(value != nullptr, "aclCreateTensor failed");
        values_[size_++] = value;
    }
    void Destroy()
    {
        for (size_t i = 0; i < size_; ++i) {
            aclDestroyTensor(values_[i]);
            values_[i] = nullptr;
        }
        size_ = 0;
    }
    aclTensor* operator[](size_t index) const { return values_[index]; }

private:
    std::array<aclTensor*, kTensorCount> values_{};
    size_t size_ = 0;
};

using Outputs = std::tuple<at::Tensor, at::Tensor>;

// Same path at::native::empty_npu takes, minus the dispatcher round trip; the
// TensorImpl inherits its device from the caching allocator's block.
at::Tensor EmptyOutput(const at::Tensor& like)
{
    c10::Allocator* allocator = c10_npu::NPUCachingAllocator::get();
    const c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
    return at::Tensor(at::detail::empty_generic(like.sizes(), allocator, keys,
                                                like.scalar_type(), at::MemoryFormat::Contiguous));
}

void RecordInputs(const std::vector<at::Tensor>& inputs, const c10_npu::NPUStream& stream)
{
    for (const auto& input : inputs) {
        c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(), stream);
    }
}

// aclnn's two-phase API returns an executor precisely so that callers may reuse
// it. Reuse is only valid when every operand still describes the same device
// buffers, so the key below covers all seven data pointers plus shape, dtype,
// device and epsilon; any change forces a rebuild.
struct InvocationKey {
    c10::DeviceIndex device = -1;
    at::ScalarType dtype = at::kHalf;
    int64_t n = 0;
    int64_t h = 0;
    double epsilon = 0.0;
    std::array<const void*, kTensorCount> data{};

    bool operator==(const InvocationKey& other) const
    {
        return device == other.device && dtype == other.dtype && n == other.n && h == other.h &&
               epsilon == other.epsilon && data == other.data;
    }
};

struct Invocation {
    bool valid = false;
    InvocationKey key;
    std::array<aclTensor*, kTensorCount> tensors{};
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    void Destroy()
    {
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
            executor = nullptr;
        }
        for (aclTensor*& tensor : tensors) {
            if (tensor != nullptr) {
                aclDestroyTensor(tensor);
                tensor = nullptr;
            }
        }
        valid = false;
        workspaceSize = 0;
    }
    // The destructor deliberately does not touch the runtime: the cached handles
    // are released by release_cache() before the interpreter tears ACL down.
    ~Invocation() = default;
};

// Allocation order for equal-sized output blocks is stable in practice but not
// guaranteed, so keep a few address keys instead of exactly one.
constexpr size_t kCacheSlots = 4;

struct InvocationTable {
    std::array<Invocation, kCacheSlots> slots;
    size_t next = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;

    Invocation* Find(const InvocationKey& key)
    {
        for (Invocation& slot : slots) {
            if (slot.valid && slot.key == key) {
                return &slot;
            }
        }
        return nullptr;
    }
    Invocation& NextVictim()
    {
        Invocation& slot = slots[next];
        next = (next + 1) % kCacheSlots;
        return slot;
    }
    size_t ActiveCount() const
    {
        size_t count = 0;
        for (const Invocation& slot : slots) {
            if (slot.valid) {
                ++count;
            }
        }
        return count;
    }
    void Destroy()
    {
        for (Invocation& slot : slots) {
            slot.Destroy();
        }
        next = 0;
    }
};

std::mutex& InvocationMutex()
{
    static std::mutex mutex;
    return mutex;
}

InvocationTable& InvocationCache()
{
    static InvocationTable table;
    return table;
}

enum class CacheMode { kOff, kAuto, kForce };

CacheMode ReadCacheMode()
{
    static const CacheMode mode = [] {
        const char* raw = std::getenv("DUAL_AFFINE_LAYERNORM_EXECUTOR_CACHE");
        if (raw == nullptr || raw[0] == '\0') {
            return CacheMode::kAuto;
        }
        if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "off") == 0) {
            return CacheMode::kOff;
        }
        if (std::strcmp(raw, "force") == 0) {
            return CacheMode::kForce;
        }
        return CacheMode::kAuto;
    }();
    return mode;
}

using SetRepeatableFn = aclError (*)(aclOpExecutor*);

SetRepeatableFn RepeatableSetter()
{
    static const SetRepeatableFn setter = [] {
        // torch_npu loads the ACL runtime into a local scope, so RTLD_DEFAULT does
        // not always see its symbols. Promote the already-linked library first.
        const char* libraries[] = {"libascendcl.so", "libascendcl.so.1", "libnnopbase.so"};
        for (const char* name : libraries) {
            void* handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                continue;
            }
            void* symbol = dlsym(handle, "aclSetAclOpExecutorRepeatable");
            if (symbol != nullptr) {
                return reinterpret_cast<SetRepeatableFn>(symbol);
            }
        }
        return reinterpret_cast<SetRepeatableFn>(dlsym(RTLD_DEFAULT, "aclSetAclOpExecutorRepeatable"));
    }();
    return setter;
}

InvocationKey MakeKey(const Plan& plan, const std::vector<at::Tensor>& inputs,
                      const at::Tensor& y1, const at::Tensor& y2)
{
    InvocationKey key;
    key.device = plan.device;
    key.dtype = plan.dtype;
    key.n = plan.n;
    key.h = plan.h;
    key.epsilon = plan.epsilon;
    key.data = {inputs[0].data_ptr(), inputs[1].data_ptr(), inputs[2].data_ptr(),
                inputs[3].data_ptr(), inputs[4].data_ptr(), y1.data_ptr(), y2.data_ptr()};
    return key;
}

// Returns false when the cached form cannot be used and the caller must fall
// back to a per-call executor.
bool RunAclnnReusing(const Plan& plan, const std::vector<at::Tensor>& inputs,
                     const at::Tensor& y1, const at::Tensor& y2,
                     const c10_npu::NPUStream& stream)
{
    const CacheMode mode = ReadCacheMode();
    if (mode == CacheMode::kOff) {
        return false;
    }
    const InvocationKey key = MakeKey(plan, inputs, y1, y2);
    std::lock_guard<std::mutex> lock(InvocationMutex());
    InvocationTable& table = InvocationCache();
    if (Invocation* hit = table.Find(key)) {
        table.hits += 1;
        TORCH_CHECK(aclnnDualAffineLayerNorm(nullptr, hit->workspaceSize, hit->executor,
                                             stream.stream()) == ACL_SUCCESS,
                    "aclnnDualAffineLayerNorm launch failed");
        return true;
    }
    if (mode == CacheMode::kAuto && RepeatableSetter() == nullptr) {
        return false;
    }
    table.misses += 1;
    Invocation& invocation = table.NextVictim();
    invocation.Destroy();
    invocation.key = key;
    const std::array<at::Tensor, kTensorCount> operands{
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], y1, y2};
    for (size_t i = 0; i < kTensorCount; ++i) {
        invocation.tensors[i] = CreateAclTensor(operands[i]);
        TORCH_CHECK(invocation.tensors[i] != nullptr, "aclCreateTensor failed");
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const aclError status = aclnnDualAffineLayerNormGetWorkspaceSize(
        invocation.tensors[0], invocation.tensors[1], invocation.tensors[2],
        invocation.tensors[3], invocation.tensors[4], plan.epsilon,
        invocation.tensors[5], invocation.tensors[6], &workspaceSize, &executor);
    if (status != ACL_SUCCESS) {
        invocation.Destroy();
        TORCH_CHECK(false, "aclnnDualAffineLayerNormGetWorkspaceSize failed: ", status);
    }
    TORCH_CHECK(workspaceSize == 0, "explicit workspaceSize != 0");
    const SetRepeatableFn setter = RepeatableSetter();
    if (setter != nullptr && setter(executor) != ACL_SUCCESS) {
        aclDestroyAclOpExecutor(executor);
        invocation.Destroy();
        return false;
    }
    invocation.executor = executor;
    invocation.workspaceSize = workspaceSize;
    invocation.valid = true;
    TORCH_CHECK(aclnnDualAffineLayerNorm(nullptr, workspaceSize, executor, stream.stream()) == ACL_SUCCESS,
                "aclnnDualAffineLayerNorm launch failed");
    return true;
}

Outputs RunAclnnPerCall(const Plan& plan, const std::vector<at::Tensor>& inputs,
                        const at::Tensor& y1, const at::Tensor& y2,
                        const c10_npu::NPUStream& stream)
{
    AclTensorSet tensors;
    tensors.Add(inputs[0]);
    tensors.Add(inputs[1]);
    tensors.Add(inputs[2]);
    tensors.Add(inputs[3]);
    tensors.Add(inputs[4]);
    tensors.Add(y1);
    tensors.Add(y2);
    uint64_t workspaceSize = 0;
    aclOpExecutor* rawExecutor = nullptr;
    const aclError status = aclnnDualAffineLayerNormGetWorkspaceSize(
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], plan.epsilon,
        tensors[5], tensors[6], &workspaceSize, &rawExecutor);
    TORCH_CHECK(status == ACL_SUCCESS, "aclnnDualAffineLayerNormGetWorkspaceSize failed: ", status);
    std::unique_ptr<aclOpExecutor, decltype(&aclDestroyAclOpExecutor)> executor(
        rawExecutor, aclDestroyAclOpExecutor);
    TORCH_CHECK(workspaceSize == 0, "explicit workspaceSize != 0");
    TORCH_CHECK(aclnnDualAffineLayerNorm(nullptr, workspaceSize, executor.get(), stream.stream()) == ACL_SUCCESS,
                "aclnnDualAffineLayerNorm launch failed");
    return {y1, y2};
}

Outputs RunAclnn(const Plan& plan, const std::vector<at::Tensor>& inputs)
{
    c10::DeviceGuard deviceGuard(inputs[0].device());
    auto stream = c10_npu::getCurrentNPUStream(inputs[0].device().index());
    auto y1 = EmptyOutput(inputs[0]);
    auto y2 = EmptyOutput(inputs[0]);
    // Two live caching-allocator blocks never share a data pointer, so this is
    // the whole independence check; touching Storage would add refcount churn.
    TORCH_CHECK(y1.data_ptr() != y2.data_ptr(), "outputs must have independent storage");
    if (RunAclnnReusing(plan, inputs, y1, y2, stream)) {
        RecordInputs(inputs, stream);
        return {y1, y2};
    }
    Outputs outputs = RunAclnnPerCall(plan, inputs, y1, y2, stream);
    RecordInputs(inputs, stream);
    return outputs;
}

Outputs RunDirect(const Plan& plan, const std::vector<at::Tensor>& inputs)
{
    c10::DeviceGuard deviceGuard(inputs[0].device());
    auto stream = c10_npu::getCurrentNPUStream(inputs[0].device().index());
    auto y1 = EmptyOutput(inputs[0]);
    auto y2 = EmptyOutput(inputs[0]);
    // Two live caching-allocator blocks never share a data pointer, so this is
    // the whole independence check; touching Storage would add refcount churn.
    TORCH_CHECK(y1.data_ptr() != y2.data_ptr(), "outputs must have independent storage");
    LaunchDualAffineLayerNormDirect(
        plan.dtype == at::kBFloat16 ? 1 : 0, plan.n, plan.h, plan.alignedH,
        plan.rowsPerCore, plan.blockDim, plan.coresPerRow, plan.sliceH,
        static_cast<float>(plan.epsilon),
        inputs[0].data_ptr(), inputs[1].data_ptr(), inputs[2].data_ptr(),
        inputs[3].data_ptr(), inputs[4].data_ptr(), y1.data_ptr(), y2.data_ptr(), stream.stream());
    RecordInputs(inputs, stream);
    return {y1, y2};
}

pybind11::capsule Prepare(const std::vector<at::Tensor>& inputs, double epsilon)
{
    auto* plan = new Plan(MakePlan(inputs, epsilon));
    return pybind11::capsule(plan, kPlanCapsuleName, [](void* pointer) {
        delete static_cast<Plan*>(pointer);
    });
}

const Plan& FromCapsule(const pybind11::capsule& capsule)
{
    const char* name = capsule.name();
    TORCH_CHECK(name != nullptr && std::string(name) == kPlanCapsuleName,
                "invalid DualAffineLayerNorm plan capsule");
    auto* plan = static_cast<const Plan*>(capsule.get_pointer());
    TORCH_CHECK(plan != nullptr, "empty DualAffineLayerNorm plan capsule");
    return *plan;
}

std::vector<at::Tensor> ValidatePrepared(const Plan& plan, const std::vector<at::Tensor>& inputs)
{
    Plan actual = MakePlan(inputs, plan.epsilon);
    CheckAgainstPlan(actual, plan);
    return inputs;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("prepare", &Prepare, pybind11::arg("inputs"), pybind11::arg("epsilon"));
    module.def("run_aclnn", [](const std::vector<at::Tensor>& inputs, double epsilon) {
        return RunAclnn(MakePlan(inputs, epsilon), inputs);
    });
    module.def("run_direct", [](const std::vector<at::Tensor>& inputs, double epsilon) {
        return RunDirect(MakePlan(inputs, epsilon), inputs);
    });
    module.def("run_aclnn_prepared", [](const pybind11::capsule& capsule,
                                        const std::vector<at::Tensor>& inputs) {
        const auto& plan = FromCapsule(capsule);
        return RunAclnn(plan, ValidatePrepared(plan, inputs));
    });
    module.def("run_direct_prepared", [](const pybind11::capsule& capsule,
                                         const std::vector<at::Tensor>& inputs) {
        const auto& plan = FromCapsule(capsule);
        return RunDirect(plan, ValidatePrepared(plan, inputs));
    });
    module.def("release_cache", [] {
        std::lock_guard<std::mutex> lock(InvocationMutex());
        InvocationCache().Destroy();
    });
    module.def("cache_status", [] {
        std::lock_guard<std::mutex> lock(InvocationMutex());
        const InvocationTable& table = InvocationCache();
        pybind11::dict status;
        const char* mode = "auto";
        if (ReadCacheMode() == CacheMode::kOff) {
            mode = "off";
        } else if (ReadCacheMode() == CacheMode::kForce) {
            mode = "force";
        }
        status["mode"] = mode;
        status["repeatable_symbol"] = RepeatableSetter() != nullptr;
        status["active"] = table.ActiveCount() != 0;
        status["slots"] = static_cast<int>(table.ActiveCount());
        status["hits"] = table.hits;
        status["misses"] = table.misses;
        return status;
    });
}