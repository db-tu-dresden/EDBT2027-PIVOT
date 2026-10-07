/* cvt_u32_f32_128 — source idiom (SSE2): u32 -> f32 from the two 16-bit halves.
   hi * 65536 is exact, so the final add rounds once, like a direct conversion. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "cvt_u32_f32_128"
#define NTO1_CATEGORY  "conversion"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline __m128 cvt_u32_f32(__m128i v) {
    __m128i lo  = _mm_and_si128(v, _mm_set1_epi32(0xffff));
    __m128i hi  = _mm_srli_epi32(v, 16);
    __m128  fhi = _mm_mul_ps(_mm_cvtepi32_ps(hi), _mm_set1_ps(65536.0f));
    return _mm_add_ps(fhi, _mm_cvtepi32_ps(lo));
}

NTO1_RUN_UNARY(__m128i, __m128, cvt_u32_f32)
