/* hmax_s32_128 — clang_nto1: one horizontal max reduction. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "hmax_s32_128"
#define NTO1_CATEGORY  "horizontal_reduction"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 0

static inline int32_t hmax_s32(v4i32 v) {
    return __builtin_reduce_max(v);
}

NTO1_RUN_REDUCE(v4i32, hmax_s32)
