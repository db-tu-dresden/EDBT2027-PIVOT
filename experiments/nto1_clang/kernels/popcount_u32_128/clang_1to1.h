/* popcount_u32_128 — clang_1to1: source idiom intrinsic by intrinsic; PSHUFB,
   PMADDUBSW and PMADDWD are emulated (common/clang_vec.h). */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "popcount_u32_128"
#define NTO1_CATEGORY  "popcount"
#define NTO1_DIRECTION "x86->arm"
#define NTO1_VARIANT   "clang_1to1"
#define NTO1_IN_STRIDE  16
#define NTO1_OUT_STRIDE 16

static inline v4u32 popcnt_u32(v4u32 v) {
    const v16u8 lut = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};     /* _mm_setr_epi8 */
    const v16u8 m   = (v16u8)0x0f;                            /* _mm_set1_epi8 */
    v16u8 lo  = (v16u8)v & m;                                 /* _mm_and_si128 */
    v16u8 hi  = (v16u8)((v8u16)v >> 4) & m;                   /* _mm_and_si128(_mm_srli_epi16) */
    v16u8 cnt = emu_shuffle_epi8(lut, lo)
              + emu_shuffle_epi8(lut, hi);                    /* _mm_add_epi8(_mm_shuffle_epi8 x2) */
    v8i16 s16 = emu_maddubs_epi16(cnt, (v16i8)1);             /* _mm_maddubs_epi16 */
    return (v4u32)emu_madd_epi16(s16, (v8i16)1);              /* _mm_madd_epi16 */
}

NTO1_RUN_UNARY(v4u32, v4u32, popcnt_u32)
