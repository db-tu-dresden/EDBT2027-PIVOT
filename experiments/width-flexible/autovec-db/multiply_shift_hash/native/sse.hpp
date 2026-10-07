// SSE (128-bit) multiply-shift hash over an arbitrary-length buffer: the 128-bit
// kernel of autovec-db/benchmarks/hashing.cpp (struct x86_128_hash), generalized
// from a fixed 64-key array to a runtime-length buffer. SSE has no 64-bit vector
// multiply, so it is emulated with 32-bit ops.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

struct sse_hash {
  using VecT = __m128i;
  static constexpr std::size_t KEYS_PER_VEC = sizeof(VecT) / sizeof(uint64_t);

  static __m128i multiply(__m128i a, __m128i b) {
    // logic same as in https://github.com/vectorclass/version2/blob/master/vectori128.h#L4062-L4081
    const __m128i zero = _mm_setzero_si128();
    const __m128i b_hi_lo_swapped = _mm_shuffle_epi32(b, 0xB1);
    const __m128i product_hi_lo_pairs = _mm_mullo_epi32(a, b_hi_lo_swapped);
    const __m128i hi_lo_pair_product_sums = _mm_hadd_epi32(product_hi_lo_pairs, zero);
    const __m128i product_hi_lo_pairs_shuffled = _mm_shuffle_epi32(hi_lo_pair_product_sums, 0x73);
    const __m128i product_lo_lo = _mm_mul_epu32(a, b);
    return _mm_add_epi64(product_lo_lo, product_hi_lo_pairs_shuffled);
  }

  void operator()(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
    const auto* typed_input_ptr = reinterpret_cast<const VecT*>(in);
    auto* typed_output_ptr = reinterpret_cast<VecT*>(out);

    const VecT factor_vec = _mm_set1_epi64x(MULTIPLY_CONSTANT);
    const int shift = 64 - required_bits;

    for (std::size_t i = 0; i < n / KEYS_PER_VEC; ++i) {
      const auto multiplied = multiply(typed_input_ptr[i], factor_vec);
      const auto shifted = _mm_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }
  }
};

static inline void pivot_hash(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
  sse_hash{}(in, out, n, required_bits);
}
