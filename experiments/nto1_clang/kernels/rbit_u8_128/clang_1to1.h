/* rbit_u8_128 — clang_1to1: the SSSE3 nibble-reverse LUT idiom translated
   intrinsic by intrinsic; each PSHUFB is emulated lane by lane. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "rbit_u8_128"
#define NTO1_CATEGORY  "fused_special"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v16u8 rbit_u8(v16u8 v) {
    const v16u8 rev = {0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,
                       0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF};    /* _mm_setr_epi8 */
    const v16u8 m = (v16u8)0x0f;                            /* _mm_set1_epi8 */
    v16u8 lo = v & m;                                       /* _mm_and_si128 */
    v16u8 hi = (v16u8)((v8u16)v >> 4) & m;                  /* _mm_and_si128(_mm_srli_epi16(v, 4), m) */
    return (v16u8)((v8u16)emu_shuffle_epi8(rev, lo) << 4)   /* _mm_or_si128(_mm_slli_epi16(_mm_shuffle_epi8, 4), */
         | emu_shuffle_epi8(rev, hi);                       /*              _mm_shuffle_epi8) */
}

NTO1_RUN_UNARY(v16u8, v16u8, rbit_u8)
