/* neon_popcount_u32_128 — source idiom (NEON): byte CNT, then two widening pairwise adds. */
#include <arm_neon.h>
#include "nto1_loops.h"

#define NTO1_ID        "neon_popcount_u32_128"
#define NTO1_CATEGORY  "popcount"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline uint32x4_t popcnt_u32(uint32x4_t v) {
    return vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(v))));
}

NTO1_RUN_UNARY(uint32x4_t, uint32x4_t, popcnt_u32)
