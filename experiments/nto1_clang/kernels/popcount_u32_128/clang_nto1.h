/* popcount_u32_128 — clang_nto1: one elementwise popcount on 32-bit lanes. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "popcount_u32_128"
#define NTO1_CATEGORY  "popcount"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v4u32 popcnt_u32(v4u32 v) {
    return __builtin_elementwise_popcount(v);
}

NTO1_RUN_UNARY(v4u32, v4u32, popcnt_u32)
