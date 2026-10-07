// Native x86 baseline @256-bit (hand-written AVX2), buffer form.
// 64-bit vector multiply emulated with 32-bit ops (AVX2 has no 64-bit vec mul).
#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

static inline __m256i ms_multiply256(__m256i a, __m256i b) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i b_hi_lo_swapped = _mm256_shuffle_epi32(b, 0xB1);
  const __m256i product_hi_lo_pairs = _mm256_mullo_epi32(a, b_hi_lo_swapped);
  const __m256i hi_lo_pair_product_sums = _mm256_hadd_epi32(product_hi_lo_pairs, zero);
  const __m256i product_hi_lo_pairs_shuffled = _mm256_shuffle_epi32(hi_lo_pair_product_sums, 0x73);
  const __m256i product_lo_lo = _mm256_mul_epu32(a, b);
  return _mm256_add_epi64(product_lo_lo, product_hi_lo_pairs_shuffled);
}

static inline void pivot_hash(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
  const auto* ip = reinterpret_cast<const __m256i*>(in);
  auto* op = reinterpret_cast<__m256i*>(out);
  const __m256i factor = _mm256_set1_epi64x(MULTIPLY_CONSTANT);
  const int shift = 64 - required_bits;
  constexpr std::size_t per = sizeof(__m256i) / sizeof(uint64_t);
  for (std::size_t i = 0; i < n / per; ++i)
    op[i] = _mm256_srli_epi64(ms_multiply256(ip[i], factor), shift);
}
