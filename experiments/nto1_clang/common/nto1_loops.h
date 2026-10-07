/* nto1_loops.h — the benchmark loop shapes, shared by every variant of a kernel
 * so that the variants differ only in the per-op function FN: memcpy loads, then
 * elementwise kernels store to `out` (the driver folds `out` once, outside the
 * timed region) and reductions add their scalar result to the returned sum. */
#ifndef NTO1_LOOPS_H
#define NTO1_LOOPS_H
#include <stdint.h>
#include <string.h>
#include "nto1_common.h"

/* Hook for asm_count.py, which disables unrolling of the benchmark loop to read
   one op per iteration; empty in the benchmark binaries. */
#ifndef NTO1_LOOP_PRAGMA
#define NTO1_LOOP_PRAGMA
#endif

/* one 16-byte input -> one 16-byte result (IN_STRIDE 16, OUT_STRIDE 16) */
#define NTO1_RUN_UNARY(TIN, TOUT, FN)                                         \
    static uint64_t nto1_run(const uint8_t *in, uint8_t *out, size_t nops) {  \
        NTO1_LOOP_PRAGMA                                                      \
        for (size_t i = 0; i < nops; ++i) {                                   \
            TIN v; memcpy(&v, in + i * 16, 16);                               \
            TOUT r = FN(v);                                                   \
            memcpy(out + i * 16, &r, 16);                                     \
        }                                                                     \
        return 0;                                                             \
    }

/* two 16-byte inputs -> one 16-byte result (IN_STRIDE 32, OUT_STRIDE 16) */
#define NTO1_RUN_BINARY(TIN, TOUT, FN)                                        \
    static uint64_t nto1_run(const uint8_t *in, uint8_t *out, size_t nops) {  \
        NTO1_LOOP_PRAGMA                                                      \
        for (size_t i = 0; i < nops; ++i) {                                   \
            TIN a; memcpy(&a, in + i * 32, 16);                               \
            TIN b; memcpy(&b, in + i * 32 + 16, 16);                          \
            TOUT r = FN(a, b);                                                \
            memcpy(out + i * 16, &r, 16);                                     \
        }                                                                     \
        return 0;                                                             \
    }

/* one 16-byte input -> integer scalar (IN_STRIDE 16, OUT_STRIDE 0) */
#define NTO1_RUN_REDUCE(TIN, FN)                                              \
    static uint64_t nto1_run(const uint8_t *in, uint8_t *out, size_t nops) {  \
        (void)out;                                                            \
        uint64_t c = 0;                                                       \
        NTO1_LOOP_PRAGMA                                                      \
        for (size_t i = 0; i < nops; ++i) {                                   \
            TIN v; memcpy(&v, in + i * 16, 16);                               \
            c += (uint32_t)FN(v);                                             \
        }                                                                     \
        return c;                                                             \
    }

/* one 16-byte input -> float scalar, folded by its bit pattern */
#define NTO1_RUN_REDUCE_F32(TIN, FN)                                          \
    static uint64_t nto1_run(const uint8_t *in, uint8_t *out, size_t nops) {  \
        (void)out;                                                            \
        uint64_t c = 0;                                                       \
        NTO1_LOOP_PRAGMA                                                      \
        for (size_t i = 0; i < nops; ++i) {                                   \
            TIN v; memcpy(&v, in + i * 16, 16);                               \
            float r = FN(v);                                                  \
            uint32_t bits; memcpy(&bits, &r, 4);                              \
            c += bits;                                                        \
        }                                                                     \
        return c;                                                             \
    }

#endif
