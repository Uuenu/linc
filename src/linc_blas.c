/* Level-2 and level-3 kernels: gemv, ger, trsv and four GEMM algorithms.
 *
 * The GEMM variants form a progression. Each one fixes the main bottleneck
 * of the previous one; bench/bench.c measures the effect of every step.
 *
 *   NAIVE      C[i][j] = sum_k A[i][k] * B[k][j]. The inner loop walks B down
 *              a column: stride n, a cache miss per element for large n.
 *   REORDERED  Same arithmetic in i-k-j order. The inner loop is now
 *              C[i][:] += A[i][k] * B[k][:]: both streams unit-stride and
 *              trivially vectorised. Still re-reads all of B for each row
 *              of C, so it falls off once B no longer fits in cache.
 *   BLOCKED    Tiles all three loops so that the working set of each tile
 *              triple stays in cache and is reused many times.
 *   PACKED     The BLIS/GotoBLAS structure: copy ("pack") blocks of A and B
 *              into contiguous panels laid out in exactly the order the
 *              micro-kernel consumes them, then compute a small MR x NR block
 *              of C entirely in registers. Packing costs O(n^2) and buys
 *              streaming, TLB-friendly access for the O(n^3) part.
 */
#include "linc_internal.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Level 2                                                                   */
/* ------------------------------------------------------------------------ */

linc_status linc_gemv(linc_trans trans, double alpha, const linc_mat *A,
                      const linc_vec *x, double beta, linc_vec *y)
{
    if (!A || !x || !y)
        return LINC_ERR_NULL;
    size_t m = A->rows, n = A->cols;
    size_t xn = trans == LINC_TRANS ? m : n;
    size_t yn = trans == LINC_TRANS ? n : m;
    if (x->n != xn || y->n != yn)
        return LINC_ERR_DIM;
    if (linc_overlaps(y->data, y->n, x->data, x->n) ||
        linc_overlaps(y->data, y->n, A->data, linc_mat_size(A)))
        return LINC_ERR_ALIAS;

    if (trans == LINC_NO_TRANS) {
        /* y_i = alpha * <row i of A, x> + beta * y_i: one dot per row. */
        for (size_t i = 0; i < m; i++) {
            double s = linc__dot(&LINC_AT(A, i, 0), x->data, n);
            y->data[i] = beta == 0.0 ? alpha * s : alpha * s + beta * y->data[i];
        }
    } else {
        /* y = alpha * A^T x is a combination of the rows of A. Walking A
         * row by row keeps the access contiguous (a column walk would not). */
        if (beta == 0.0)
            linc_vec_fill(y, 0.0);
        else if (beta != 1.0)
            linc_vec_scale(y, beta);
        for (size_t i = 0; i < m; i++) {
            const double *a = &LINC_AT(A, i, 0);
            double s = alpha * x->data[i];
            if (s == 0.0)
                continue;
            for (size_t j = 0; j < n; j++)
                y->data[j] += s * a[j];
        }
    }
    return LINC_OK;
}

linc_status linc_ger(double alpha, const linc_vec *x, const linc_vec *y,
                     linc_mat *A)
{
    if (!x || !y || !A)
        return LINC_ERR_NULL;
    if (x->n != A->rows || y->n != A->cols)
        return LINC_ERR_DIM;
    if (linc_overlaps(A->data, linc_mat_size(A), x->data, x->n) ||
        linc_overlaps(A->data, linc_mat_size(A), y->data, y->n))
        return LINC_ERR_ALIAS;
    for (size_t i = 0; i < A->rows; i++) {
        double s = alpha * x->data[i];
        double *a = &LINC_AT(A, i, 0);
        for (size_t j = 0; j < A->cols; j++)
            a[j] += s * y->data[j];
    }
    return LINC_OK;
}

linc_status linc_trsv(linc_uplo uplo, linc_trans trans, linc_diag diag,
                      const linc_mat *T, linc_vec *b)
{
    if (!T || !b)
        return LINC_ERR_NULL;
    if (T->rows != T->cols || b->n != T->rows)
        return LINC_ERR_DIM;
    if (linc_overlaps(b->data, b->n, T->data, linc_mat_size(T)))
        return LINC_ERR_ALIAS;
    size_t n = T->rows;
    double *x = b->data;
    int unit = diag == LINC_UNIT;

    /* Check the diagonal first so that b is untouched on failure. */
    if (!unit)
        for (size_t i = 0; i < n; i++)
            if (LINC_AT(T, i, i) == 0.0)
                return LINC_ERR_SINGULAR;

    /* Row-major storage makes rows contiguous. The no-transpose cases use
     * the "dot product" form (row i of T times solved part of x). The
     * transposed cases would need columns of T for that, so they use the
     * equivalent "axpy" form instead: once x_i is known, subtract its
     * contribution (row i of T) from all remaining right-hand sides. */
    if (uplo == LINC_LOWER && trans == LINC_NO_TRANS) {
        for (size_t i = 0; i < n; i++) {
            const double *t = &LINC_AT(T, i, 0);
            double s = x[i] - linc__dot(t, x, i);
            x[i] = unit ? s : s / t[i];
        }
    } else if (uplo == LINC_UPPER && trans == LINC_NO_TRANS) {
        for (size_t i = n; i-- > 0;) {
            const double *t = &LINC_AT(T, i, 0);
            double s = x[i] - linc__dot(t + i + 1, x + i + 1, n - i - 1);
            x[i] = unit ? s : s / t[i];
        }
    } else if (uplo == LINC_LOWER) { /* L^T is upper: go backwards */
        for (size_t i = n; i-- > 0;) {
            const double *t = &LINC_AT(T, i, 0);
            if (!unit)
                x[i] /= t[i];
            double xi = x[i];
            for (size_t k = 0; k < i; k++)
                x[k] -= t[k] * xi;
        }
    } else { /* U^T is lower: go forwards */
        for (size_t i = 0; i < n; i++) {
            const double *t = &LINC_AT(T, i, 0);
            if (!unit)
                x[i] /= t[i];
            double xi = x[i];
            for (size_t k = i + 1; k < n; k++)
                x[k] -= t[k] * xi;
        }
    }
    return LINC_OK;
}

/* ------------------------------------------------------------------------ */
/* GEMM                                                                      */
/* ------------------------------------------------------------------------ */

/* C <- beta * C, with beta == 0 meaning "overwrite" (C may hold NaNs). */
static void scale_c(linc_mat *C, double beta)
{
    if (beta == 0.0)
        linc_mat_fill(C, 0.0);
    else if (beta != 1.0)
        linc_mat_scale(C, beta);
}

static void gemm_naive(size_t m, size_t n, size_t k, double alpha,
                       const double *A, const double *B, double beta,
                       double *C)
{
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t p = 0; p < k; p++)
                s += A[i * k + p] * B[p * n + j]; /* B: stride-n access */
            C[i * n + j] = beta == 0.0 ? alpha * s : alpha * s + beta * C[i * n + j];
        }
}

/* Assumes C was already scaled by beta. */
static void gemm_reordered(size_t m, size_t n, size_t k, double alpha,
                           const double *restrict A, const double *restrict B,
                           double *restrict C)
{
    for (size_t i = 0; i < m; i++) {
        double *restrict c = C + i * n;
        for (size_t p = 0; p < k; p++) {
            double a = alpha * A[i * k + p];
            const double *restrict b = B + p * n;
            for (size_t j = 0; j < n; j++)
                c[j] += a * b[j];
        }
    }
}

#ifndef LINC_BLK_M
#define LINC_BLK_M 64
#endif
#ifndef LINC_BLK_K
#define LINC_BLK_K 128
#endif
#ifndef LINC_BLK_N
#define LINC_BLK_N 256
#endif

/* Assumes C was already scaled by beta. A BLK_K x BLK_N tile of B (256 KiB
 * with the defaults) is reused for BLK_M rows of C before moving on. */
static void gemm_blocked(size_t m, size_t n, size_t k, double alpha,
                         const double *restrict A, const double *restrict B,
                         double *restrict C)
{
    for (size_t jj = 0; jj < n; jj += LINC_BLK_N) {
        size_t je = jj + LINC_BLK_N < n ? jj + LINC_BLK_N : n;
        for (size_t kk = 0; kk < k; kk += LINC_BLK_K) {
            size_t ke = kk + LINC_BLK_K < k ? kk + LINC_BLK_K : k;
            for (size_t ii = 0; ii < m; ii += LINC_BLK_M) {
                size_t ie = ii + LINC_BLK_M < m ? ii + LINC_BLK_M : m;
                for (size_t i = ii; i < ie; i++) {
                    double *restrict c = C + i * n;
                    for (size_t p = kk; p < ke; p++) {
                        double a = alpha * A[i * k + p];
                        const double *restrict b = B + p * n;
                        for (size_t j = jj; j < je; j++)
                            c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

/* Packed GEMM parameters.
 *   MR x NR  register block of C computed by the micro-kernel. 4 x 8 doubles
 *            is 32 accumulators: 16 NEON or 8 AVX2 registers.
 *   KC       depth of a packed panel; an MR x KC sliver of A plus a KC x NR
 *            sliver of B should sit in L1.
 *   MC       rows of the packed A block (MC x KC), sized for L2.
 *   NC       columns of the packed B block (KC x NC), sized for L3/L2. */
#ifndef LINC_MR
#define LINC_MR 4
#endif
#ifndef LINC_NR
#define LINC_NR 8
#endif
#ifndef LINC_KC
#define LINC_KC 256
#endif
#ifndef LINC_MC
#define LINC_MC 128
#endif
#ifndef LINC_NC
#define LINC_NC 2048
#endif

#define MR LINC_MR
#define NR LINC_NR

static size_t min_sz(size_t a, size_t b) { return a < b ? a : b; }

/* Pack an mc x kc block of A (row stride lda) into MR-row panels:
 * panel r holds, for p = 0..kc-1, the MR values A[r*MR + 0..MR-1][p].
 * Rows past mc are zero-filled so the micro-kernel never needs bounds
 * checks. alpha is folded in here, where it costs O(mc * kc) not O(mnk). */
static void pack_a(size_t mc, size_t kc, double alpha, const double *A,
                   size_t lda, double *restrict out)
{
    for (size_t ir = 0; ir < mc; ir += MR) {
        size_t mr = min_sz(MR, mc - ir);
        for (size_t p = 0; p < kc; p++) {
            for (size_t i = 0; i < mr; i++)
                out[i] = alpha * A[(ir + i) * lda + p];
            for (size_t i = mr; i < MR; i++)
                out[i] = 0.0;
            out += MR;
        }
    }
}

/* Pack a kc x nc block of B into NR-column panels: panel r holds, for each
 * p, the NR values B[p][r*NR + 0..NR-1]. Zero-padded like pack_a. */
static void pack_b(size_t kc, size_t nc, const double *B, size_t ldb,
                   double *restrict out)
{
    for (size_t jr = 0; jr < nc; jr += NR) {
        size_t nr = min_sz(NR, nc - jr);
        for (size_t p = 0; p < kc; p++) {
            const double *b = B + p * ldb + jr;
            for (size_t j = 0; j < nr; j++)
                out[j] = b[j];
            for (size_t j = nr; j < NR; j++)
                out[j] = 0.0;
            out += NR;
        }
    }
}

/* GCC and Clang vector extensions: one source that compiles to NEON, SSE or
 * AVX. Left to itself the compiler keeps a 4 x 8 accumulator block in 32
 * scalar registers (and spills), which runs no faster than the reordered
 * loop; spelling out the vectors is what makes the packed kernel pay off. */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(LINC_NO_VECTOR_EXT)
#if defined(__AVX__)
#define VLEN 4
#else
#define VLEN 2
#endif
typedef double vdouble __attribute__((vector_size(VLEN * sizeof(double))));
#define LINC_HAVE_VECTOR_EXT 1
#if NR % VLEN != 0
#error "LINC_NR must be a multiple of the SIMD width"
#endif
#endif

/* The micro-kernel: C[0..mr)[0..nr) += sum_p a_p * b_p^T, a sequence of kc
 * rank-1 updates on an MR x NR block held in registers. Per step p it loads
 * NR values of B once, broadcasts each of the MR values of A, and issues
 * MR * NR / VLEN fused multiply-adds: no stores, no reloads of C. */
static void micro_kernel(size_t kc, const double *restrict a,
                         const double *restrict b, double *restrict C,
                         size_t ldc, size_t mr, size_t nr)
{
    double acc[MR][NR];
#ifdef LINC_HAVE_VECTOR_EXT
    enum { NV = NR / VLEN };
    vdouble vacc[MR][NV];
    for (size_t i = 0; i < MR; i++)
        for (size_t j = 0; j < NV; j++)
            vacc[i][j] = (vdouble){0};
    for (size_t p = 0; p < kc; p++) {
        vdouble bv[NV];
        for (size_t j = 0; j < NV; j++)
            memcpy(&bv[j], b + j * VLEN, sizeof bv[j]); /* unaligned load */
        for (size_t i = 0; i < MR; i++) {
            vdouble av;
            for (size_t l = 0; l < VLEN; l++)
                av[l] = a[i]; /* broadcast */
            for (size_t j = 0; j < NV; j++)
                vacc[i][j] += av * bv[j];
        }
        a += MR;
        b += NR;
    }
    memcpy(acc, vacc, sizeof acc); /* same row-major layout */
#else
    memset(acc, 0, sizeof acc);
    for (size_t p = 0; p < kc; p++) {
        for (size_t i = 0; i < MR; i++)
            for (size_t j = 0; j < NR; j++)
                acc[i][j] += a[i] * b[j];
        a += MR;
        b += NR;
    }
#endif
    if (mr == MR && nr == NR) {
        for (size_t i = 0; i < MR; i++)
            for (size_t j = 0; j < NR; j++)
                C[i * ldc + j] += acc[i][j];
    } else {
        for (size_t i = 0; i < mr; i++)
            for (size_t j = 0; j < nr; j++)
                C[i * ldc + j] += acc[i][j];
    }
}

/* Assumes C was already scaled by beta. Returns 0 if scratch allocation
 * failed (C is untouched in that case). */
static int gemm_packed(size_t m, size_t n, size_t k, double alpha,
                       const double *A, const double *B, double *C)
{
    size_t kc_max = min_sz(LINC_KC, k);
    size_t mc_max = (min_sz(LINC_MC, m) + MR - 1) / MR * MR;
    size_t nc_max = (min_sz(LINC_NC, n) + NR - 1) / NR * NR;
    double *Ap = malloc(mc_max * kc_max * sizeof(double));
    double *Bp = malloc(kc_max * nc_max * sizeof(double));
    if (!Ap || !Bp) {
        free(Ap);
        free(Bp);
        return 0;
    }

    for (size_t jc = 0; jc < n; jc += LINC_NC) {
        size_t nc = min_sz(LINC_NC, n - jc);
        for (size_t pc = 0; pc < k; pc += LINC_KC) {
            size_t kc = min_sz(LINC_KC, k - pc);
            pack_b(kc, nc, B + pc * n + jc, n, Bp);
            for (size_t ic = 0; ic < m; ic += LINC_MC) {
                size_t mc = min_sz(LINC_MC, m - ic);
                pack_a(mc, kc, alpha, A + ic * k + pc, k, Ap);
                for (size_t jr = 0; jr < nc; jr += NR) {
                    size_t nr = min_sz(NR, nc - jr);
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = min_sz(MR, mc - ir);
                        micro_kernel(kc, Ap + ir * kc, Bp + jr * kc,
                                     C + (ic + ir) * n + jc + jr, n, mr, nr);
                    }
                }
            }
        }
    }
    free(Ap);
    free(Bp);
    return 1;
}

/* Below this many multiply-adds, packing overhead outweighs its benefit. */
#define LINC_PACK_THRESHOLD (64u * 64u * 64u)

linc_status linc_gemm_ex(linc_gemm_algo algo, double alpha, const linc_mat *A,
                         const linc_mat *B, double beta, linc_mat *C)
{
    if (!A || !B || !C)
        return LINC_ERR_NULL;
    size_t m = A->rows, k = A->cols, n = B->cols;
    if (B->rows != k || C->rows != m || C->cols != n)
        return LINC_ERR_DIM;
    if (linc_mats_overlap(C, A) || linc_mats_overlap(C, B))
        return LINC_ERR_ALIAS;
    if (algo < LINC_GEMM_AUTO || algo > LINC_GEMM_PACKED)
        return LINC_ERR_ARG;

    if (algo == LINC_GEMM_NAIVE) {
        gemm_naive(m, n, k, alpha, A->data, B->data, beta, C->data);
        return LINC_OK;
    }

    /* Volume m*n*k in floating point so huge shapes cannot overflow. */
    double volume = (double)m * (double)n * (double)k;
    if (algo == LINC_GEMM_AUTO)
        algo = volume < LINC_PACK_THRESHOLD ? LINC_GEMM_REORDERED
                                            : LINC_GEMM_PACKED;

    scale_c(C, beta);
    if (alpha == 0.0 || k == 0)
        return LINC_OK;
    if (algo == LINC_GEMM_REORDERED)
        gemm_reordered(m, n, k, alpha, A->data, B->data, C->data);
    else if (algo == LINC_GEMM_BLOCKED ||
             !gemm_packed(m, n, k, alpha, A->data, B->data, C->data))
        /* Also the fallback when packing scratch cannot be allocated: the
         * blocked kernel needs no memory, so GEMM itself never fails. */
        gemm_blocked(m, n, k, alpha, A->data, B->data, C->data);
    return LINC_OK;
}

linc_status linc_gemm(double alpha, const linc_mat *A, const linc_mat *B,
                      double beta, linc_mat *C)
{
    return linc_gemm_ex(LINC_GEMM_AUTO, alpha, A, B, beta, C);
}

linc_status linc_mat_mul(const linc_mat *A, const linc_mat *B, linc_mat *C)
{
    return linc_gemm_ex(LINC_GEMM_AUTO, 1.0, A, B, 0.0, C);
}
