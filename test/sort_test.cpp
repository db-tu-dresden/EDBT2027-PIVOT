// Checks SortSVE::Sort (from whichever sortSVE.hpp is on the include path) against std::sort.
#include "sortSVE.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

template <class T>
int check(const char* name) {
  std::mt19937_64 rng(7);
  int failures = 0;
  for (std::size_t n : {1, 2, 3, 7, 16, 31, 64, 100, 255, 1000, 4097, 20000}) {
    for (int dist = 0; dist < 3; ++dist) {  // 0: random, 1: few distinct values, 2: reverse sorted
      std::vector<T> v(n);
      for (std::size_t i = 0; i < n; ++i)
        v[i] = dist == 0 ? T(int64_t(rng() % 2000000001) - 1000000000) / (sizeof(T) == 8 ? T(3) : T(1))
             : dist == 1 ? T(rng() % 5)
                         : T(n - i);
      std::vector<T> ref = v;
      std::sort(ref.begin(), ref.end());
      SortSVE::Sort<T, size_t>(v.data(), n);
      if (v != ref && failures++ < 5) std::printf("  %s MISMATCH n=%zu dist=%d\n", name, n, dist);
    }
  }
  std::printf("  %s: %s\n", name, failures ? "FAIL" : "OK");
  return failures;
}

int main() {
  int f = check<int>("int32") + check<double>("double");
  std::printf(f ? "FAIL\n" : "OK\n");
  return f != 0;
}
