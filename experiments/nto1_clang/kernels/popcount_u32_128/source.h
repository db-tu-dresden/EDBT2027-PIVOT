/* popcount_u32_128 — source idiom (SSSE3): nibble-LUT byte popcount, then
   PMADDUBSW + PMADDWD against ones to sum the four bytes of each lane. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "popcount_u32_128"
#define NTO1_CATEGORY  "popcount"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline __m128i popcnt_u32(__m128i v) {
    const __m128i lut = _mm_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m128i m   = _mm_set1_epi8(0x0f);
    __m128i lo  = _mm_and_si128(v, m);
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(v, 4), m);
    __m128i cnt = _mm_add_epi8(_mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
    __m128i s16 = _mm_maddubs_epi16(cnt, _mm_set1_epi8(1));
    return _mm_madd_epi16(s16, _mm_set1_epi16(1));
}

NTO1_RUN_UNARY(__m128i, __m128i, popcnt_u32)
