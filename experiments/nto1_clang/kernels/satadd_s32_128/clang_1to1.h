/* satadd_s32_128 — clang_1to1: the SSE2 overflow-blend saturating add
   translated intrinsic by intrinsic. Vector integer + wraps in Clang, like
   PADDD. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "satadd_s32_128"
#define NTO1_CATEGORY  "saturating"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline v4i32 qadd_s32(v4i32 a, v4i32 b) {
    v4i32 s   = a + b;                                /* _mm_add_epi32 */
    v4i32 ov  = (a ^ s) & (b ^ s);                    /* _mm_and_si128(_mm_xor_si128, _mm_xor_si128) */
    v4i32 m   = ov >> 31;                             /* _mm_srai_epi32 */
    v4i32 sat = (a >> 31) ^ (v4i32)0x7fffffff;        /* _mm_xor_si128(_mm_srai_epi32(a, 31), _mm_set1_epi32) */
    return (m & sat) | (~m & s);                      /* _mm_or_si128(_mm_and_si128, _mm_andnot_si128) */
}

NTO1_RUN_BINARY(v4i32, v4i32, qadd_s32)
