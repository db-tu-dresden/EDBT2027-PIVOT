/* cvt_u32_f32_128 — clang_1to1: source idiom intrinsic by intrinsic. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "cvt_u32_f32_128"
#define NTO1_CATEGORY  "conversion"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v4f32 cvt_u32_f32(v4u32 v) {
    v4u32 lo  = v & (v4u32)0xffff;                                   /* _mm_and_si128 */
    v4u32 hi  = v >> 16;                                             /* _mm_srli_epi32 */
    v4f32 fhi = __builtin_convertvector((v4i32)hi, v4f32)
              * (v4f32)65536.0f;                                     /* _mm_mul_ps(_mm_cvtepi32_ps(hi), ..) */
    return fhi + __builtin_convertvector((v4i32)lo, v4f32);          /* _mm_add_ps(fhi, _mm_cvtepi32_ps(lo)) */
}

NTO1_RUN_UNARY(v4u32, v4f32, cvt_u32_f32)
