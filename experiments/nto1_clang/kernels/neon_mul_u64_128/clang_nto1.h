/* neon_mul_u64_128 — clang_nto1: one 64-bit lane multiply. */
#include "clang_vec.h"
#include "nto1_loops.h"

#define NTO1_ID        "neon_mul_u64_128"
#define NTO1_CATEGORY  "multiply"
#define NTO1_DIRECTION "arm->x86"
#define NTO1_VARIANT   "clang_nto1"
#define NTO1_IN_STRIDE  32
#define NTO1_OUT_STRIDE 16

static inline v2u64 mul_u64(v2u64 a, v2u64 b) {
    return a * b;
}

NTO1_RUN_BINARY(v2u64, v2u64, mul_u64)
