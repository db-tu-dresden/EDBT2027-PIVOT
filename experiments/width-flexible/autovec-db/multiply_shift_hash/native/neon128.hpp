// Native ARM baseline @128-bit: upstream autovec-db's hand-written NEON kernel (neon_hash in
// benchmarks/hashing.cpp), buffer form. 64-bit vector multiply emulated with 32-bit ops (NEON
// has no 64-bit vec mul). Upstream relies on lax vector conversions; the vreinterpretq casts
// spell those out and change no instruction.
#pragma once
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

static inline uint64x2_t ms_multiply_neon(uint64x2_t a, uint64x2_t b) {
  const uint32x4_t b_hi_lo_swapped = vrev64q_u32(vreinterpretq_u32_u64(b));
  const uint32x4_t product_hi_lo_pairs = vmulq_u32(vreinterpretq_u32_u64(a), b_hi_lo_swapped);
  const uint64x2_t hi_lo_pair_product_sums = vpaddlq_u32(product_hi_lo_pairs);
  const uint64x2_t hi_lo_pair_product_sums_shifted = vshlq_n_u64(hi_lo_pair_product_sums, 32);
  const uint32x2_t a_lo = vmovn_u64(a);
  const uint32x2_t b_lo = vmovn_u64(b);
  return vmlal_u32(hi_lo_pair_product_sums_shifted, b_lo, a_lo);
}

static inline void pivot_hash(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
  const auto* ip = reinterpret_cast<const uint64x2_t*>(in);
  auto* op = reinterpret_cast<uint64x2_t*>(out);
  const uint64x2_t factor = vdupq_n_u64(MULTIPLY_CONSTANT);
  // NEON has no right shift by a run-time amount, so shift left by the negated amount.
  const int64x2_t shift_vec = vdupq_n_s64(-(64 - required_bits));
  constexpr std::size_t per = sizeof(uint64x2_t) / sizeof(uint64_t);
  for (std::size_t i = 0; i < n / per; ++i)
    op[i] = vshlq_u64(ms_multiply_neon(ip[i], factor), shift_vec);
}
