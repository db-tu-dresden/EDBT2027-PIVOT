/* neon_clz_u64_128 — source idiom (NEON has no 64-bit CLZ): 32-bit CLZ of both halves,
   low-half count added where the high half is zero. */
#include <arm_neon.h>
#include "nto1_loops.h"

#define NTO1_ID        "neon_clz_u64_128"
#define NTO1_CATEGORY  "clz"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "source"
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

static inline uint64x2_t clz_u64(uint64x2_t v) {
    uint64x2_t c    = vreinterpretq_u64_u32(vclzq_u32(vreinterpretq_u32_u64(v)));
    uint64x2_t hi   = vshrq_n_u64(c, 32);
    uint64x2_t lo   = vandq_u64(c, vdupq_n_u64(0xffffffffu));
    uint64x2_t full = vceqq_u64(hi, vdupq_n_u64(32));
    return vaddq_u64(hi, vandq_u64(full, lo));
}

NTO1_RUN_UNARY(uint64x2_t, uint64x2_t, clz_u64)
