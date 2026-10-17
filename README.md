# linc

A small, readable linear algebra library in C11, written to be studied and
to be used.

The code is meant to be read: every algorithm says *why* it is written the
way it is, not only what it computes. It is also meant to be used: there is
one header, no dependencies beyond libm, and no global state. Errors come back
as status codes, overlapping buffers are detected, and the tests run clean
under AddressSanitizer and UBSan. The GEMM kernel reaches roughly 80% of one
core's FP64 peak.

```c
#include <linc.h>

double a[] = {2, 1, -1,   -3, -1, 2,   -2, 1, 2};
double b[] = {8, -11, -3}, x[3];
linc_mat A  = linc_mat_view(a, 3, 3);
linc_vec bv = linc_vec_view(b, 3), xv = linc_vec_view(x, 3);

linc_status st = linc_solve(&A, &bv, &xv);   /* x = [2, 3, -1] */
if (st != LINC_OK)
    fprintf(stderr, "solve: %s\n", linc_status_str(st));
```

## What is in it

| Area | Functions | Algorithm |
|---|---|---|
| Level 1 (vectors) | `dot`, `axpy`, `scale`, `add`, `sub`, `norm1`, `norm2`, `norm_inf` | multi-accumulator dot; overflow-safe `norm2` (LAPACK `dlassq` style) |
| Level 2 (matrix-vector) | `gemv` (± transpose), `ger`, `trsv` (all uplo/trans/diag) | row-major-friendly dot and axpy forms |
| Level 3 (matrix-matrix) | `gemm`, `gemm_ex`, `mat_mul` | 4 algorithms, from naive to BLIS-style packed SIMD |
| Matrix ops | `transpose`, `add`, `sub`, `scale`, `trace`, `norm` (1/∞/Frobenius/max), `is_symmetric` | cache-tiled transpose |
| LU | `lu_factor`, `lu_solve`, `lu_solve_mat`, `lu_det`, `lu_inverse` | Gaussian elimination, partial pivoting |
| Cholesky | `cholesky`, `cholesky_solve` | Cholesky–Banachiewicz; works in place |
| QR | `qr_factor`, `qr_q`, `qr_r`, `qr_solve` | Householder reflections, LAPACK `dgeqrf` storage |
| Eigen | `eig_sym` | cyclic Jacobi rotations; sorted values plus orthonormal vectors |
| One-shot | `solve`, `det`, `inverse`, `lstsq` | factor, use, free |

Everything is declared, with its contract documented, in
[`include/linc.h`](include/linc.h).

## Building

```sh
make                 # build/liblinc.a and build/liblinc.{so,dylib}
make test            # 20k+ checks
make sanitize        # the same tests under ASan + UBSan
make examples        # build/solve_system, build/least_squares, build/pca
make bench NATIVE=1  # benchmarks tuned for this CPU
make bench NATIVE=1 BLAS=1   # also compare with Accelerate / OpenBLAS
sudo make install    # PREFIX=/usr/local by default
```

Or with CMake (exports `linc::linc` for `find_package(linc)`):

```sh
cmake -S . -B build -DLINC_NATIVE=ON
cmake --build build && ctest --test-dir build
```

Tested with Apple clang 21 on macOS arm64 and GCC 14 on Linux x86-64.

## Design

- **Dense, row-major `double`.** Element `(i, j)` is `data[i * cols + j]`,
  accessed with `LINC_AT(m, i, j)`. That is the layout C programmers already
  use, so `linc_mat_view` can wrap an existing array without copying.
- **Status codes, never aborts.** Every fallible function returns
  `linc_status`, and `linc_status_str` turns it into text. The library never
  prints, exits, or asserts.
- **The caller owns the outputs.** Results go into objects you allocated,
  so you control memory and can reuse buffers in hot loops. The only internal
  allocations are scratch space for factorizations and GEMM packing.
- **Aliasing is checked.** For example, `gemm(A, B, C)` with `C` overlapping
  `A` returns `LINC_ERR_ALIAS` instead of silently computing garbage.
  Operations that are safe in place (`lu_solve`, `cholesky`, `mat_add`, square
  `transpose`) say so in the header.
- **Overflow-safe allocation.** `rows * cols * sizeof(double)` is checked
  before allocating, and data is 64-byte aligned.
- **No global state**, so it is thread-safe on distinct objects.
- **BLAS semantics where they exist.** With `beta == 0`, `gemm` and `gemv`
  do not read `C`/`y`, so NaN garbage in an uninitialised output never
  leaks into the result.

### Numerical notes

- **LU** uses partial pivoting, so every multiplier satisfies |l| ≤ 1. A
  pivot with |u_kk| ≤ n·ε·max|A| marks the matrix as numerically singular.
  The factorization still completes (so `det` still works), but `solve` and
  `inverse` return `LINC_ERR_SINGULAR`. This catches matrices such as
  `[[1,2,3],[4,5,6],[7,8,9]]`, whose computed pivot is ~1e-16 rather than
  exactly 0.
- **QR least squares** returns `LINC_ERR_RANK_DEFICIENT` when some
  |r_kk| ≤ m·ε·max|r_ii|. Prefer `lstsq` over the normal equations, which
  square the condition number.
- **Cholesky** reads only the lower triangle and doubles as the cheapest
  positive-definiteness test (`LINC_ERR_NOT_SPD`, which also catches NaN).
- **Jacobi** gives small eigenvalues to high relative accuracy and stops
  once off(A) ≤ ε·‖A‖_F. It is practical up to a few hundred rows; LAPACK's
  tridiagonal QR is faster for large n.

## Benchmarks

Apple M3 Pro, single core, Apple clang 21, `make bench NATIVE=1 BLAS=1`.
Each figure is the best of N runs, with each measurement lasting at least
0.25 s. Full output: [`bench/results-m3pro.txt`](bench/results-m3pro.txt).

### GEMM: what each optimization buys (GFLOP/s, higher is better)

| n | naive (ijk) | reordered (ikj) | blocked | **packed** | speedup packed vs naive | system BLAS¹ |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4.2 | 18.3 | 18.3 | **43.5** | 10× | 299 |
| 128 | 3.0 | 16.9 | 16.9 | **45.2** | 15× | 356 |
| 256 | 2.3 | 14.3 | 14.2 | **49.0** | 21× | 383 |
| 512 | 1.4 | 14.2 | 11.7 | **47.4** | 34× | 379 |
| 1024 | 1.3 | 13.9 | 11.2 | **49.0** | 37× | 388 |
| 2048 | — | 12.7 | 10.3 | **37.8–47** | — | 330 |

¹ Apple Accelerate. It uses the M-series AMX matrix coprocessor and
multiple threads, so treat it as a ceiling rather than a competitor. One
NEON core peaks at about 60 GFLOP/s in FP64, so the packed kernel reaches
about 80% of what plain C can get from a core.

Reading the table:

1. **Naive → reordered (5–10×).** The `ijk` loop walks `B` down a column
   with stride `n`, so every access misses cache, and it gets *worse* as n
   grows. Swapping to `ikj` makes the inner loop `C[i,:] += a * B[k,:]`,
   which is unit-stride and vectorizes.
2. **Reordered → blocked (≈0×).** Plain cache tiling barely helps here. The
   M3's large caches and hardware prefetchers already hide most of the
   streaming cost of the `ikj` loop, and the inner loop is still bound by
   loads and stores to `C`. Tiling alone is not enough.
3. **Blocked → packed (3–4×).** This step matters. Copying blocks of `A`
   and `B` into contiguous micro-panels lets a 4×8 block of `C` live
   entirely in **registers** for the whole `k` loop: each step is 4 loads and
   16 vector FMAs, with no traffic to `C`. One trap: written as a scalar
   `acc[4][8]`, clang emitted 32 *scalar* FMAs and ran no faster than
   `ikj`. Writing the kernel with explicit SIMD vectors (GCC/Clang vector
   extensions, which compile to NEON, SSE, or AVX) is what made the 3.5×
   gain.

`linc_gemm` (`LINC_GEMM_AUTO`) uses the reordered kernel below 64³
multiply-adds, where packing overhead dominates, and the packed kernel
above that. If the packing buffers cannot be allocated, it falls back to
the allocation-free blocked kernel, so GEMM never fails for lack of memory.

### Decompositions (GFLOP/s, standard flop counts)

| n | LU | solve | inverse | QR | lstsq | Cholesky |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10.1 | 8.3 | 11.5 | 14.9 | 13.4 | 7.5 |
| 250 | 9.7 | 8.9 | 9.7 | 11.7 | 11.3 | 9.1 |
| 500 | 10.0 | 9.5 | 10.0 | 11.8 | 11.3 | 10.2 |
| 1000 | 10.2 | 10.0 | 10.0 | 10.8 | 11.5 | 10.5 |

These run at about 10 GFLOP/s compared with GEMM's 49, and that gap is the
next lesson. LU, QR, and Cholesky here are the classic *unblocked*
algorithms, made of level-2 operations (one row or column at a time), so
they are limited by memory bandwidth. LAPACK reorganizes them into blocked
forms that spend most of their time in GEMM; that is the natural next step
(see the roadmap below).

Cholesky in particular went from 2.3 to 10.5 GFLOP/s just by computing its
inner dot products with eight independent accumulators instead of one.
Without `-ffast-math` the compiler is not allowed to reassociate a
floating-point sum, so a single accumulator serializes every add on the FMA
latency. The same fix sped up `gemv` 3.5× and `norm2` 5×.

### Level 1/2 and the eigen solver

| operation | n | time | throughput |
|---|---:|---:|---:|
| dot | 1,000,000 | 176 µs | 11.4 GFLOP/s |
| axpy | 1,000,000 | 187 µs | 10.7 GFLOP/s |
| norm2 (overflow-safe) | 1,000,000 | 136 µs | 14.7 GFLOP/s |
| gemv | 4096 | 3.5 ms | 9.5 GFLOP/s |
| gemv transposed | 4096 | 2.4 ms | 14.2 GFLOP/s |
| transpose | 1024 | 0.66 ms | 25.5 GB/s |
| eig_sym (values + vectors) | 100 | 4.7 ms | |
| eig_sym (values + vectors) | 400 | 625 ms | |

The transpose tile is 8×8, not the usual 32×32. With power-of-two row
strides, every row of a tile maps to the same cache set, so 32×32 tiles
thrash the set associativity; 8×8 measured 2–5× faster for n ≥ 1024.

Run `build/bench_linc --help` for options (`--quick`, `--csv`, `--only gemm`,
`--min-time S`).

## Reading order

If you are here to learn, read the sources in this order:

1. [`src/linc_core.c`](src/linc_core.c): allocation, level-1 kernels, and
   why `norm2` is not just `sqrt(dot(x, x))`.
2. [`src/linc_blas.c`](src/linc_blas.c): the GEMM progression from naive to
   packed, and triangular solves in row-major order.
3. [`src/linc_lu.c`](src/linc_lu.c): Gaussian elimination and why pivoting
   is not optional.
4. [`src/linc_chol.c`](src/linc_chol.c): the SPD special case at half the
   cost.
5. [`src/linc_qr.c`](src/linc_qr.c): Householder reflections and least
   squares.
6. [`src/linc_eig.c`](src/linc_eig.c): Jacobi rotations.

Then run the [examples](examples/): solving a system, fitting a polynomial,
and PCA through the eigendecomposition of a covariance matrix.

## Testing

`tests/test_linc.c` checks each decomposition through its defining identity
(PA = LU, QᵀQ = I, QR = A, LLᵀ = A, AV = VΛ) on random matrices of many
shapes, including sizes that straddle every GEMM block and tile boundary.
It also covers hand-computed cases, error paths (dimension, alias,
singular, not-SPD, rank-deficient), overflow and underflow in `norm2`, and
NaN handling for `beta == 0`. CI runs the suite with GCC and Clang on Linux
and macOS, under sanitizers, and through CMake.

## Limitations and roadmap

- `double` only, dense only, single-threaded.
- LU, QR, and Cholesky are unblocked (level-2). Blocked versions built on
  `gemm` would bring them close to GEMM speed.
- Not yet implemented: SVD (one-sided Jacobi would fit the style),
  nonsymmetric eigenproblems, condition-number estimation, and a
  `float` variant.

## License

MIT; see [LICENSE](LICENSE).
