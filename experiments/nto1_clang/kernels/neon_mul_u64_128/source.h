/* neon_mul_u64_128 — source idiom (NEON has no 64-bit lane multiply): cross products by
   VMULQ_U32 on swapped halves, summed and shifted, plus UMLAL of the low halves. */
#include <arm_neon.h>
#include "nto1_loops.h"

#define NTO1_ID        "neon_mul_u64_128"
#define NTO1_CATEGORY  "multiply"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline uint64x2_t mul_u64(uint64x2_t a, uint64x2_t b) {
    uint32x2_t a_lo  = vmovn_u64(a);
    uint32x2_t b_lo  = vmovn_u64(b);
    uint32x4_t b_swp = vrev64q_u32(vreinterpretq_u32_u64(b));
    uint32x4_t cross = vmulq_u32(b_swp, vreinterpretq_u32_u64(a));
    uint64x2_t hi    = vshlq_n_u64(vpaddlq_u32(cross), 32);
    return vmlal_u32(hi, a_lo, b_lo);
}

NTO1_RUN_BINARY(uint64x2_t, uint64x2_t, mul_u64)
