/* satadd_s32_128 — clang_nto1: one elementwise saturating add. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "satadd_s32_128"
#define NTO1_CATEGORY  "saturating"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline v4i32 qadd_s32(v4i32 a, v4i32 b) {
    return __builtin_elementwise_add_sat(a, b);
}

NTO1_RUN_BINARY(v4i32, v4i32, qadd_s32)
