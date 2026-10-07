/* neon_popcount_u32_128 — clang_1to1: source idiom intrinsic by intrinsic; UADDLP is emulated. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "neon_popcount_u32_128"
#define NTO1_CATEGORY  "popcount"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v4u32 popcnt_u32(v4u32 v) {
    v16u8 c8 = __builtin_elementwise_popcount((v16u8)v);    /* vcntq_u8 */
    return emu_vpaddlq_u16(emu_vpaddlq_u8(c8));             /* vpaddlq_u16(vpaddlq_u8) */
}

NTO1_RUN_UNARY(v4u32, v4u32, popcnt_u32)
