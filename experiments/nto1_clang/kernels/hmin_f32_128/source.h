/* hmin_f32_128 — source idiom (SSE): SHUFPS/MINPS ladder. */
#include <immintrin.h>
#include "nto1_loops.h"

#define NTO1_ID        "hmin_f32_128"
#define NTO1_CATEGORY  "horizontal_reduction"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "source"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 0
#define NTO1_HAS_PREP   1

static inline void nto1_prep(uint8_t *in, size_t nops) {
    size_t words = nops * 4;
    for (size_t i = 0; i < words; ++i) {
        uint32_t w; memcpy(&w, in + i * 4, 4);
        w = (w & 0x807fffffu) | 0x40000000u;
        memcpy(in + i * 4, &w, 4);
    }
}

static inline float hmin_f32(__m128 v) {
    __m128 t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
    v = _mm_min_ps(v, t);
    t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,0,3,2));
    v = _mm_min_ps(v, t);
    return _mm_cvtss_f32(v);
}

NTO1_RUN_REDUCE_F32(__m128, hmin_f32)
