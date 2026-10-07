/* satadd_s32_128 — source idiom (SSE2): overflow-blend saturating add. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "satadd_s32_128"
#define NTO1_CATEGORY  "saturating"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline __m128i qadd_s32(__m128i a, __m128i b) {
    __m128i s  = _mm_add_epi32(a, b);
    __m128i ov = _mm_and_si128(_mm_xor_si128(a, s), _mm_xor_si128(b, s));
    __m128i m  = _mm_srai_epi32(ov, 31);
    __m128i sat = _mm_xor_si128(_mm_srai_epi32(a, 31), _mm_set1_epi32(0x7fffffff));
    return _mm_or_si128(_mm_and_si128(m, sat), _mm_andnot_si128(m, s));
}

NTO1_RUN_BINARY(__m128i, __m128i, qadd_s32)
