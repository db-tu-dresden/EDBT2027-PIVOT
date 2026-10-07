/* neon_mul_u64_128 — clang_1to1: source idiom intrinsic by intrinsic; UADDLP and UMLAL are emulated. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "neon_mul_u64_128"
#define NTO1_CATEGORY  "multiply"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline v2u64 mul_u64(v2u64 a, v2u64 b) {
    v2u32 a_lo  = __builtin_convertvector(a, v2u32);                        /* vmovn_u64 */
    v2u32 b_lo  = __builtin_convertvector(b, v2u32);                        /* vmovn_u64 */
    v4u32 b_swp = __builtin_shufflevector((v4u32)b, (v4u32)b, 1, 0, 3, 2);  /* vrev64q_u32 */
    v4u32 cross = b_swp * (v4u32)a;                                         /* vmulq_u32 */
    v2u64 hi    = emu_vpaddlq_u32(cross) << 32;                             /* vshlq_n_u64(vpaddlq_u32) */
    return emu_vmlal_u32(hi, a_lo, b_lo);                                   /* vmlal_u32 */
}

NTO1_RUN_BINARY(v2u64, v2u64, mul_u64)
