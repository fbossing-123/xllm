#ifndef DUAL_AFFINE_LAYER_NORM_SAMPLE_COMMON_H
#define DUAL_AFFINE_LAYER_NORM_SAMPLE_COMMON_H

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void CheckAcl(int status, const char* call)
{
    if (status != 0) {
        throw std::runtime_error(std::string(call) + " failed: " + std::to_string(status));
    }
}

inline aclDataType ParseDtype(const std::string& value)
{
    if (value == "float16") return ACL_FLOAT16;
    if (value == "bfloat16") return ACL_BF16;
    throw std::invalid_argument("dtype must be float16 or bfloat16");
}

inline std::vector<char> ReadBinary(const std::string& path, size_t bytes)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("wrong input size: " + path);
    }
    std::vector<char> data(bytes);
    input.seekg(0);
    if (bytes != 0 && !input.read(data.data(), bytes)) {
        throw std::runtime_error("read failed: " + path);
    }
    return data;
}

class SampleContext {
public:
    SampleContext(int device) : device_(device)
    {
        try {
            CheckAcl(aclInit(nullptr), "aclInit");
            initialized_ = true;
            CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice");
            deviceSet_ = true;
            CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
        } catch (...) {
            Release();
            throw;
        }
    }
    ~SampleContext() { Release(); }
    aclrtStream stream() const { return stream_; }
    int device() const { return device_; }
private:
    void Release()
    {
        if (stream_ != nullptr) aclrtDestroyStream(stream_);
        stream_ = nullptr;
        if (deviceSet_) aclrtResetDevice(device_);
        deviceSet_ = false;
        if (initialized_) aclFinalize();
        initialized_ = false;
    }
    int device_ = 0;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class SampleTensor {
public:
    SampleTensor(const std::string& directory, const std::string& name,
                 std::vector<int64_t> shape, aclDataType dtype, bool input)
        : shape_(std::move(shape)), bytes_(2 * ElementCount(shape_))
    {
        try {
            host_ = input ? ReadBinary(directory + "/" + name + ".bin", bytes_)
                          : std::vector<char>(bytes_, 0);
            CheckAcl(aclrtMalloc(&device_, std::max<size_t>(bytes_, 1), ACL_MEM_MALLOC_HUGE_FIRST),
                     "aclrtMalloc");
            if (input) {
                CheckAcl(aclrtMemcpy(device_, bytes_, host_.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE),
                         "aclrtMemcpy H2D");
            }
            std::vector<int64_t> strides(shape_.size(), 1);
            for (size_t i = shape_.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape_[i - 1];
            tensor_ = aclCreateTensor(shape_.data(), shape_.size(), dtype, strides.data(), 0,
                                      ACL_FORMAT_ND, shape_.data(), shape_.size(), device_);
            if (tensor_ == nullptr) throw std::runtime_error("aclCreateTensor failed");
        } catch (...) {
            Release();
            throw;
        }
    }
    ~SampleTensor() { Release(); }
    SampleTensor(const SampleTensor&) = delete;
    SampleTensor& operator=(const SampleTensor&) = delete;
    aclTensor* tensor() const { return tensor_; }
    void* data() const { return device_; }
    void Save(const std::string& directory, const std::string& name)
    {
        CheckAcl(aclrtMemcpy(host_.data(), bytes_, device_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy D2H");
        std::ofstream output(directory + "/" + name + ".bin", std::ios::binary);
        if (!output || (bytes_ != 0 && !output.write(host_.data(), bytes_))) {
            throw std::runtime_error("write failed: " + name);
        }
    }
private:
    static size_t ElementCount(const std::vector<int64_t>& shape)
    {
        size_t count = 1;
        for (int64_t dim : shape) count *= static_cast<size_t>(dim);
        return count;
    }
    void Release()
    {
        if (tensor_ != nullptr) aclDestroyTensor(tensor_);
        tensor_ = nullptr;
        if (device_ != nullptr) aclrtFree(device_);
        device_ = nullptr;
    }
    std::vector<int64_t> shape_;
    size_t bytes_ = 0;
    std::vector<char> host_;
    void* device_ = nullptr;
    aclTensor* tensor_ = nullptr;
};

#endif
