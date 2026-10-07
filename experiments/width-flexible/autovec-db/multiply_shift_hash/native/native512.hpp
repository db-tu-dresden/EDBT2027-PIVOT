// Native x86 baseline @512-bit (hand-written AVX-512), buffer form.
// AVX-512DQ has a real 64-bit vector multiply (_mm512_mullo_epi64), so no
// 32-bit emulation is needed here — matching hashing.cpp's x86_512_hash.
#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

static inline void pivot_hash(const uint64_t* in, uint64_t* out, std::size_t n, int required_bits) {
  const auto* ip = reinterpret_cast<const __m512i*>(in);
  auto* op = reinterpret_cast<__m512i*>(out);
  const __m512i factor = _mm512_set1_epi64(MULTIPLY_CONSTANT);
  const unsigned shift = 64 - required_bits;
  constexpr std::size_t per = sizeof(__m512i) / sizeof(uint64_t);
  for (std::size_t i = 0; i < n / per; ++i)
    op[i] = _mm512_srli_epi64(_mm512_mullo_epi64(ip[i], factor), shift);
}
