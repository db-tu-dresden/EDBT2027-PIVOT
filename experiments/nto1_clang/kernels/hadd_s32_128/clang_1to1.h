/* hadd_s32_128 — clang_1to1: the SSE2 shuffle/add ladder translated
   intrinsic by intrinsic. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "hadd_s32_128"
#define NTO1_CATEGORY  "horizontal_reduction"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 0

static inline int32_t hsum_s32(v4i32 v) {
    v4i32 t = __builtin_shufflevector(v, v, 2, 3, 0, 1);    /* _mm_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2)) */
    v = v + t;                                              /* _mm_add_epi32 */
    t = __builtin_shufflevector(v, v, 1, 0, 3, 2);          /* _mm_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1)) */
    v = v + t;                                              /* _mm_add_epi32 */
    return v[0];                                            /* _mm_cvtsi128_si32 */
}

NTO1_RUN_REDUCE(v4i32, hsum_s32)
