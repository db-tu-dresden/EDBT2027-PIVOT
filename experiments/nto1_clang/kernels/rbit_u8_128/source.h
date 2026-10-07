/* rbit_u8_128 — source idiom (SSSE3): nibble-reverse LUT via PSHUFB. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "rbit_u8_128"
#define NTO1_CATEGORY  "fused_special"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline __m128i rbit_u8(__m128i v) {
    const __m128i rev = _mm_setr_epi8(0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,
                                      0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF);
    const __m128i m = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(v, m);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m);
    return _mm_or_si128(_mm_slli_epi16(_mm_shuffle_epi8(rev, lo), 4),
                        _mm_shuffle_epi8(rev, hi));
}

NTO1_RUN_UNARY(__m128i, __m128i, rbit_u8)
