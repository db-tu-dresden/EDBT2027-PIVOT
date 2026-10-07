/* rbit_u8_128 — clang_nto1: one elementwise bit reversal. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "rbit_u8_128"
#define NTO1_CATEGORY  "fused_special"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v16u8 rbit_u8(v16u8 v) {
    return __builtin_elementwise_bitreverse(v);
}

NTO1_RUN_UNARY(v16u8, v16u8, rbit_u8)
