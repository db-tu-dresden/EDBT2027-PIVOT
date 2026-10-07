// Native x86 baseline @128-bit (hand-written SSE), buffer form.
// 64-bit vector multiply emulated with 32-bit ops (SSE has no 64-bit vec mul).
#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

static inline __m128i ms_multiply128(__m128i a, __m128i b) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i b_hi_lo_swapped = _mm_shuffle_epi32(b, 0xB1);
  const __m128i product_hi_lo_pairs = _mm_mullo_epi32(a, b_hi_lo_swapped);
  const __m128i hi_lo_pair_product_sums = _mm_hadd_epi32(product_hi_lo_pairs, zero);
  const __m128i product_hi_lo_pairs_shuffled = _mm_shuffle_epi32(hi_lo_pair_product_sums, 0x73);
  const __m128i product_lo_lo = _mm_mul_epu32(a, b);
  return _mm_add_epi64(product_lo_lo, product_hi_lo_pairs_shuffled);
}

static inline void pivot_hash(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
  const auto* ip = reinterpret_cast<const __m128i*>(in);
  auto* op = reinterpret_cast<__m128i*>(out);
  const __m128i factor = _mm_set1_epi64x(MULTIPLY_CONSTANT);
  const int shift = 64 - required_bits;
  constexpr std::size_t per = sizeof(__m128i) / sizeof(uint64_t);
  for (std::size_t i = 0; i < n / per; ++i)
    op[i] = _mm_srli_epi64(ms_multiply128(ip[i], factor), shift);
}
