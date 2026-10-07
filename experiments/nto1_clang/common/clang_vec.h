/* clang_vec.h — portable Clang vector types shared by every kernel variant.
 *
 * Only Clang vector extensions are used (ext_vector_type, operators,
 * __builtin_shufflevector, __builtin_convertvector, __builtin_elementwise_ and
 * __builtin_reduce_ builtins, vector subscripts). No target intrinsics or target builtins, so the same
 * source compiles for aarch64 and x86. Loads/stores go through memcpy. */
#ifndef NTO1_CLANG_VEC_H
#define NTO1_CLANG_VEC_H
#include <stdint.h>
#include <string.h>

typedef int8_t   v16i8  __attribute__((ext_vector_type(16)));
typedef uint8_t  v8u8   __attribute__((ext_vector_type(8)));
typedef uint8_t  v16u8  __attribute__((ext_vector_type(16)));
typedef int16_t  v4i16  __attribute__((ext_vector_type(4)));
typedef int16_t  v8i16  __attribute__((ext_vector_type(8)));
typedef int16_t  v16i16 __attribute__((ext_vector_type(16)));
typedef uint16_t v8u16  __attribute__((ext_vector_type(8)));
typedef uint16_t v16u16 __attribute__((ext_vector_type(16)));
typedef int32_t  v4i32  __attribute__((ext_vector_type(4)));
typedef int32_t  v8i32  __attribute__((ext_vector_type(8)));
typedef uint32_t v2u32  __attribute__((ext_vector_type(2)));
typedef uint32_t v4u32  __attribute__((ext_vector_type(4)));
typedef int64_t  v2i64  __attribute__((ext_vector_type(2)));
typedef uint64_t v2u64  __attribute__((ext_vector_type(2)));
typedef float    v4f32  __attribute__((ext_vector_type(4)));

typedef uint16_t v4u16  __attribute__((ext_vector_type(4)));
typedef int32_t  v16i32 __attribute__((ext_vector_type(16)));
typedef int64_t  v8i64  __attribute__((ext_vector_type(8)));
typedef double   v2f64  __attribute__((ext_vector_type(2)));
typedef _Bool    v8b    __attribute__((ext_vector_type(8)));

/* Emulations for the 1:1 variants: intrinsics that have no Clang counterpart,
 * each written as a short sequence of portable operations. */

/* _mm_shuffle_epi8 (PSHUFB): r[i] = idx[i] & 0x80 ? 0 : tbl[idx[i] & 15].
 * Clang has no dynamic byte shuffle; lane by lane, as the portable fallback of
 * PIVOT's PERMUTE lowering. */
static inline v16u8 emu_shuffle_epi8(v16u8 tbl, v16u8 idx) {
    v16u8 r;
    for (int i = 0; i < 16; ++i)
        r[i] = (idx[i] & 0x80) ? 0 : tbl[idx[i] & 15];
    return r;
}

/* _mm_maddubs_epi16 (PMADDUBSW): u8 x s8 products, adjacent pairs added with
 * signed saturation. */
static inline v8i16 emu_maddubs_epi16(v16u8 a, v16i8 b) {
    v8i16 ae = __builtin_convertvector(__builtin_shufflevector(a, a, 0, 2, 4, 6, 8, 10, 12, 14), v8i16);
    v8i16 ao = __builtin_convertvector(__builtin_shufflevector(a, a, 1, 3, 5, 7, 9, 11, 13, 15), v8i16);
    v8i16 be = __builtin_convertvector(__builtin_shufflevector(b, b, 0, 2, 4, 6, 8, 10, 12, 14), v8i16);
    v8i16 bo = __builtin_convertvector(__builtin_shufflevector(b, b, 1, 3, 5, 7, 9, 11, 13, 15), v8i16);
    return __builtin_elementwise_add_sat(ae * be, ao * bo);
}

/* _mm_madd_epi16 (PMADDWD): s16 x s16 products, adjacent pairs added (wrapping). */
static inline v4i32 emu_madd_epi16(v8i16 a, v8i16 b) {
    v8i32 p = __builtin_convertvector(a, v8i32) * __builtin_convertvector(b, v8i32);
    return __builtin_shufflevector(p, p, 0, 2, 4, 6) + __builtin_shufflevector(p, p, 1, 3, 5, 7);
}

/* vpaddlq_u8/u16/u32 (UADDLP): adjacent pairs added into lanes of twice the width. */
static inline v8u16 emu_vpaddlq_u8(v16u8 v) {
    return __builtin_convertvector(__builtin_shufflevector(v, v, 0, 2, 4, 6, 8, 10, 12, 14), v8u16)
         + __builtin_convertvector(__builtin_shufflevector(v, v, 1, 3, 5, 7, 9, 11, 13, 15), v8u16);
}
static inline v4u32 emu_vpaddlq_u16(v8u16 v) {
    return __builtin_convertvector(__builtin_shufflevector(v, v, 0, 2, 4, 6), v4u32)
         + __builtin_convertvector(__builtin_shufflevector(v, v, 1, 3, 5, 7), v4u32);
}
static inline v2u64 emu_vpaddlq_u32(v4u32 v) {
    return __builtin_convertvector(__builtin_shufflevector(v, v, 0, 2), v2u64)
         + __builtin_convertvector(__builtin_shufflevector(v, v, 1, 3), v2u64);
}

/* vmlal_u32 (UMLAL): acc + widened a * widened b. */
static inline v2u64 emu_vmlal_u32(v2u64 acc, v2u32 a, v2u32 b) {
    return acc + __builtin_convertvector(a, v2u64) * __builtin_convertvector(b, v2u64);
}

#endif
