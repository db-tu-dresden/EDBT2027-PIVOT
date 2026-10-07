# Provenance

The kernels come from **autovec-db**, `benchmarks/hashing.cpp` at commit `8bf194a`:
https://github.com/hpides/autovec-db/blob/8bf194a/benchmarks/hashing.cpp

The single kernel PIVOT actually translates is the 128-bit `struct x86_128_hash`,
extracted and generalized to a runtime-length buffer in `native/sse.hpp`. The
hand-written per-width baselines live in `native/native{128,256,512}.hpp`.
