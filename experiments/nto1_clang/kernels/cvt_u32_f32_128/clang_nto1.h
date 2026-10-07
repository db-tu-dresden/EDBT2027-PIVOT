/* cvt_u32_f32_128 — clang_nto1: one unsigned-to-float conversion. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "cvt_u32_f32_128"
#define NTO1_CATEGORY  "conversion"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v4f32 cvt_u32_f32(v4u32 v) {
    return __builtin_convertvector(v, v4f32);
}

NTO1_RUN_UNARY(v4u32, v4f32, cvt_u32_f32)
