
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DUAL_AFFINE_LAYER_NORM_H_
#define ACLNN_DUAL_AFFINE_LAYER_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDualAffineLayerNormGetWorkspaceSize
 * parameters :
 * x : required
 * gamma1 : required
 * beta1 : required
 * gamma2 : required
 * beta2 : required
 * epsilon : optional
 * y1Out : required
 * y2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDualAffineLayerNormGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma1,
    const aclTensor *beta1,
    const aclTensor *gamma2,
    const aclTensor *beta2,
    double epsilon,
    const aclTensor *y1Out,
    const aclTensor *y2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDualAffineLayerNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDualAffineLayerNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
