/* hmin_f32_128 — clang_1to1: the SSE shuffle/min ladder translated intrinsic
   by intrinsic. MINPS returns b unless a < b, hence the compare-select. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "hmin_f32_128"
#define NTO1_CATEGORY  "horizontal_reduction"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 0
#define NTO1_HAS_PREP   1

/* force each 32-bit word to a finite normal in ±[2,4) (keep sign+mantissa,
   exp=128) so min is exact and independent of NaN/±0 conventions */
static inline void nto1_prep(uint8_t *in, size_t nops) {
    size_t words = nops * 4;
    for (size_t i = 0; i < words; ++i) {
        uint32_t w; memcpy(&w, in + i * 4, 4);
        w = (w & 0x807fffffu) | 0x40000000u;
        memcpy(in + i * 4, &w, 4);
    }
}

static inline float hmin_f32(v4f32 v) {
    v4f32 t = __builtin_shufflevector(v, v, 1, 0, 7, 6);    /* _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1)) */
    v = v < t ? v : t;                                      /* _mm_min_ps */
    t = __builtin_shufflevector(v, v, 2, 3, 4, 5);          /* _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,0,3,2)) */
    v = v < t ? v : t;                                      /* _mm_min_ps */
    return v[0];                                            /* _mm_cvtss_f32 */
}

NTO1_RUN_REDUCE_F32(v4f32, hmin_f32)
