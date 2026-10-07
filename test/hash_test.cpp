// Checks pivot_hash (from whichever hash.hpp is on the include path) against scalar multiply-shift.
#include "hash.hpp"
#include <cstdio>
#include <random>
#include <vector>

int main() {
  std::mt19937_64 rng(42);
  int failures = 0;
  for (std::size_t n : {2, 4, 64, 1000, 1 << 20}) {
    for (int bits : {1, 8, 16, 32, 63}) {
      std::vector<uint64_t> in(n), out(n, 0);
      for (auto& x : in) x = rng();
      pivot_hash(in.data(), out.data(), n, bits);
      for (std::size_t i = 0; i < n; ++i) {
        uint64_t ref = (in[i] * MULTIPLY_CONSTANT) >> (64 - bits);
        if (out[i] != ref) {
          if (failures++ < 5) std::printf("MISMATCH n=%zu bits=%d i=%zu got=%llx want=%llx\n", n, bits, i,
                                          (unsigned long long)out[i], (unsigned long long)ref);
        }
      }
    }
  }
  std::printf(failures ? "FAIL (%d mismatches)\n" : "OK\n", failures);
  return failures != 0;
}
