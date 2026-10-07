/* hadd_s32_128 — source idiom (SSE2): PSHUFD/PADDD ladder. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "hadd_s32_128"
#define NTO1_CATEGORY  "horizontal_reduction"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 0

static inline int32_t hsum_s32(__m128i v) {
    __m128i t = _mm_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2));
    v = _mm_add_epi32(v, t);
    t = _mm_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1));
    v = _mm_add_epi32(v, t);
    return _mm_cvtsi128_si32(v);
}

NTO1_RUN_REDUCE(__m128i, hsum_s32)
