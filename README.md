# PIVOT

PIVOT translates SIMD intrinsics (x86 SSE/AVX/AVX-512, ARM NEON/SVE) in C/C++ into
portable SIMD code: Clang vector extensions and the Template SIMD Library (TSL)

## Setup

With Python 3.12:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

A YAML run config describes a translation job (`main.py` lists every key). The two translations
of the evaluation:

```bash
.venv/bin/python main.py run_configs/ms_hash.yaml              # multiply-shift hash, SSE -> TSL
.venv/bin/python main.py run_configs/sve_sort/tsl_fixed.yaml   # SVE quicksort -> TSL
```

`run_configs/ms_hash.yaml`:

```yaml
name: autovec_db_ms_hash
mode: single_file
source_path: experiments/width-flexible/autovec-db/multiply_shift_hash/native/sse.hpp
source_isa: x86
target_isa: tsl              # or clang_builtins
tsl_policy: fixed
tsl_width_mode: vla
output_dir: experiments/width-flexible/autovec-db/multiply_shift_hash/translated
output_name: hash.hpp
```

Each run works on a copy of the source in its own folder, `resources/logs/<name>_<stamp>/`. That
folder holds the translated file (`translated.hpp`), its companion header (`pivot_cxx_tsl.h`, or
`pivot_clang_builtins.h` for `clang_builtins`) and every intermediate stage in
`.pivot/debug/<source file>/`. Both files are also copied to the config's `output_dir`, the
translated file under `output_name`. The first run also generates the pattern graphs. Compiling
`tsl` output needs the [TSL library](https://github.com/JPietrzykTUD/tslgen-v2).

## Experiments

`experiments/` holds the results and figures of the evaluation and the kernels (except for VIP).

## License

MIT (see `LICENSE`). The third-party kernels under `experiments/` keep their own licenses; see
the `PROVENANCE.md` next to them.
