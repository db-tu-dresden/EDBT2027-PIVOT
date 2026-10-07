# sortSVE.hpp — provenance

`sortSVE.hpp` is Berenger Bramas' SVE quicksort (`SortSVE::Sort`), taken verbatim
from `external/arm-sve-sort/sortSVE.hpp`. It is the single tracked input of this
experiment:

- the `native` build compiles it directly (real SVE), and
- every PIVOT translation (`translated/…`, produced by `gen.sh`) is derived from it.

Upstream: https://gitlab.inria.fr/bramas/arm-sve-sort (`sortSVE.hpp`).

## License

`sortSVE.hpp` is distributed under its upstream MIT license:

```
The MIT License (MIT)

Copyright (c) 2017 Bramas, Berenger (bbramas)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
