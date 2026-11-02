#ifndef DUAL_AFFINE_LAYER_NORM_H
#define DUAL_AFFINE_LAYER_NORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dual_affine_layer_norm_tiling_data.h"

namespace NsDualAffineLayerNorm {

using namespace AscendC;

template <typename T>
class DualAffineLayerNorm {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma1, GM_ADDR beta1,
                                GM_ADDR gamma2, GM_ADDR beta2, GM_ADDR y1, GM_ADDR y2,
                                const DualAffineLayerNormTilingData* tiling)
    {
        n_ = tiling->n;
        h_ = tiling->h;
        alignedH_ = tiling->alignedH;
        rowsPerCore_ = tiling->rowsPerCore;
        ubBytes_ = tiling->ubBytes;
        coresPerRow_ = tiling->coresPerRow < 1 ? 1 : tiling->coresPerRow;
        sliceH_ = tiling->sliceH < 1 ? tiling->h : tiling->sliceH;
        sliceAligned_ = (sliceH_ + 63) / 64 * 64;
        epsilon_ = tiling->epsilon;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        gamma1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma1));
        beta1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta1));
        gamma2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma2));
        beta2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta2));
        y1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y1));
        y2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y2));
        pipe_.InitBuffer(rawBuf_, static_cast<uint32_t>(ubBytes_));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> storage16 = rawBuf_.Get<T>();
        LocalTensor<float> storage32 = rawBuf_.Get<float>();
        LocalTensor<T> x16 = storage16[0];
        LocalTensor<T> g1 = storage16[alignedH_];
        LocalTensor<T> b1 = storage16[alignedH_ + sliceAligned_];
        LocalTensor<T> g2 = storage16[alignedH_ + 2 * sliceAligned_];
        LocalTensor<T> b2 = storage16[alignedH_ + 3 * sliceAligned_];
        LocalTensor<float> z = storage32[(alignedH_ + 4 * sliceAligned_) / 2];
        LocalTensor<float> tmp0 = z[alignedH_];
        LocalTensor<float> tmp1 = z[2 * alignedH_];

        // With coresPerRow > 1 every core of a row recomputes the same full-row
        // statistics, so no cross-core reduction or workspace is needed.
        const int64_t block = static_cast<int64_t>(GetBlockIdx());
        int64_t firstRow = 0;
        int64_t lastRow = 0;
        int64_t start = 0;
        int64_t count = 0;
        if (coresPerRow_ > 1) {
            firstRow = block / coresPerRow_;
            lastRow = firstRow + 1;
            start = (block % coresPerRow_) * sliceH_;
            count = start < h_ ? (sliceH_ < h_ - start ? sliceH_ : h_ - start) : 0;
        } else {
            firstRow = block * rowsPerCore_;
            lastRow = firstRow + rowsPerCore_;
            start = 0;
            count = h_;
        }
        if (firstRow >= n_ || count <= 0) {
            return;
        }
        if (lastRow > n_) {
            lastRow = n_;
        }

        CopyIn(g1, gamma1Gm_, start, count);
        CopyIn(b1, beta1Gm_, start, count);
        CopyIn(g2, gamma2Gm_, start, count);
        CopyIn(b2, beta2Gm_, start, count);

        for (int64_t row = firstRow; row < lastRow; ++row) {
            CopyIn(x16, xGm_, row * h_, h_);
            SyncMte2V();

            Cast(z, x16, RoundMode::CAST_NONE, h_);
            PipeBarrier<PIPE_V>();
            ReduceSum(tmp0, z, tmp1, h_);
            const float sum = ReadScalar(tmp0);
            const float mean = sum / static_cast<float>(h_);

            Adds(z, z, -mean, h_);
            PipeBarrier<PIPE_V>();
            Mul(tmp0, z, z, h_);
            PipeBarrier<PIPE_V>();
            ReduceSum(tmp0, tmp0, tmp1, h_);
            const float squareSum = ReadScalar(tmp0);
            const float variance = squareSum / static_cast<float>(h_);
            tmp0.SetValue(0, variance + epsilon_);
            SyncScalarToVector();
            Sqrt(tmp0, tmp0, 1);
            Duplicate(tmp1, 1.0f, 1);
            PipeBarrier<PIPE_V>();
            Div(tmp0, tmp1, tmp0, 1);
            const float rstd = ReadScalar(tmp0);
            Muls(z, z, rstd, h_);
            PipeBarrier<PIPE_V>();

            LocalTensor<T> out16 = tmp1.ReinterpretCast<T>();
            ComputeBranch(g1, b1, out16, z[start], tmp0, tmp1, count);
            CopyOut(y1Gm_, out16, row * h_ + start, count);
            ComputeBranch(g2, b2, out16, z[start], tmp0, tmp1, count);
            CopyOut(y2Gm_, out16, row * h_ + start, count);
        }
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<T> local, GlobalTensor<T>& gm, int64_t offset, int64_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], params, pad);
    }

    __aicore__ inline void CopyOut(GlobalTensor<T>& gm, LocalTensor<T> local, int64_t offset, int64_t count)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(gm[offset], local, params);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncMte2V()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void SyncScalarToVector()
    {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline float ReadScalar(LocalTensor<float>& local)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return local.GetValue(0);
    }

    __aicore__ inline void ComputeBranch(LocalTensor<T> gamma, LocalTensor<T> beta,
                                         LocalTensor<T> out16, LocalTensor<float> z,
                                         LocalTensor<float> tmp0, LocalTensor<float> tmp1,
                                         int64_t count)
    {
        Cast(tmp0, gamma, RoundMode::CAST_NONE, count);
        Cast(tmp1, beta, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Mul(tmp0, z, tmp0, count);
        PipeBarrier<PIPE_V>();
        Add(tmp0, tmp0, tmp1, count);
        PipeBarrier<PIPE_V>();
        Cast(out16, tmp0, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gamma1Gm_;
    GlobalTensor<T> beta1Gm_;
    GlobalTensor<T> gamma2Gm_;
    GlobalTensor<T> beta2Gm_;
    GlobalTensor<T> y1Gm_;
    GlobalTensor<T> y2Gm_;
    int64_t n_ = 0;
    int64_t h_ = 0;
    int64_t alignedH_ = 64;
    int64_t rowsPerCore_ = 1;
    int64_t ubBytes_ = 0;
    int64_t coresPerRow_ = 1;
    int64_t sliceH_ = 1;
    int64_t sliceAligned_ = 64;
    float epsilon_ = 1e-5f;
};

} // namespace NsDualAffineLayerNorm

#endif
