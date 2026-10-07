/* neon_clz_u64_128 — clang_nto1: one elementwise count of leading zeros (0 -> 64). */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "neon_clz_u64_128"
#define NTO1_CATEGORY  "clz"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16
#define NTO1_HAS_PREP   1

/* spread the inputs over all magnitudes (w >> 0..63) so both halves of the
   idiom (upper word zero or not) are exercised */
static inline void nto1_prep(uint8_t *in, size_t nops) {
    for (size_t i = 0; i < nops * 2; ++i) {
        uint64_t w; memcpy(&w, in + i * 8, 8);
        w >>= (w >> 58) & 63;
        memcpy(in + i * 8, &w, 8);
    }
}

static inline v2u64 clz_u64(v2u64 v) {
    return __builtin_elementwise_clzg(v, (v2u64)64);
}

NTO1_RUN_UNARY(v2u64, v2u64, clz_u64)
