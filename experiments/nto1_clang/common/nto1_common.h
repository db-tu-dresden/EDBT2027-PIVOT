/* nto1_common.h — helpers shared by kernel variant headers. */
#ifndef NTO1_COMMON_H
#define NTO1_COMMON_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Fold OUT_STRIDE output bytes into a checksum. Bit-identical outputs fold to the
   same value, so this doubles as the cross-variant correctness oracle for
   elementwise kernels. */
static inline uint64_t nto1_fold_bytes(const uint8_t *o, size_t n) {
    uint64_t c = 0; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, o + i, 8); c += w; }
    for (; i < n; ++i) c += (uint64_t)o[i] << ((i & 7) * 8);
    return c;
}
#endif
