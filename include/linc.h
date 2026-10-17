/*
 * linc — a small, readable linear algebra library in C11.
 *
 * Design rules (they apply to every function in this header):
 *
 *   - Storage is dense, row-major, double precision. Element (i, j) of an
 *     r x c matrix lives at data[i * c + j]. Use LINC_AT(m, i, j).
 *   - Every fallible function returns a linc_status. Outputs are written
 *     through pointer arguments that the caller has already allocated with
 *     the right shape; on error, outputs are left in an unspecified state
 *     unless documented otherwise.
 *   - The library never prints, never aborts, and has no global state, so it
 *     is safe to call from multiple threads on distinct objects.
 *   - Functions that cannot work in place detect aliasing and return
 *     LINC_ERR_ALIAS instead of silently producing garbage.
 *
 * Objects from linc_*_new / linc_*_from / linc_*_clone are heap-allocated as
 * a single block and released with the matching linc_*_free. Objects made
 * with linc_*_view wrap caller-owned memory and must NOT be freed.
 */
#ifndef LINC_H
#define LINC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LINC_VERSION_MAJOR 0
#define LINC_VERSION_MINOR 1
#define LINC_VERSION_PATCH 0
#define LINC_VERSION "0.1.0"

/* ------------------------------------------------------------------------ */
/* Status codes                                                              */
/* ------------------------------------------------------------------------ */

typedef enum linc_status {
    LINC_OK = 0,
    LINC_ERR_NULL,           /* a required pointer argument was NULL         */
    LINC_ERR_DIM,            /* operand shapes are incompatible              */
    LINC_ERR_ALLOC,          /* memory allocation failed (or size overflow)  */
    LINC_ERR_ALIAS,          /* output overlaps an input that it must not    */
    LINC_ERR_SINGULAR,       /* matrix is (numerically) singular             */
    LINC_ERR_NOT_SPD,        /* matrix is not symmetric positive definite    */
    LINC_ERR_NOT_SYMMETRIC,  /* matrix is not symmetric                      */
    LINC_ERR_RANK_DEFICIENT, /* least-squares matrix lacks full column rank  */
    LINC_ERR_NO_CONVERGENCE, /* iterative algorithm hit its iteration limit  */
    LINC_ERR_ARG             /* some other argument is invalid               */
} linc_status;

/* Human-readable description of a status code. Never returns NULL. */
const char *linc_status_str(linc_status status);

/* ------------------------------------------------------------------------ */
/* Types                                                                     */
/* ------------------------------------------------------------------------ */

typedef struct linc_vec {
    size_t n;
    double *data;
} linc_vec;

typedef struct linc_mat {
    size_t rows;
    size_t cols;
    double *data; /* rows * cols doubles, row-major */
} linc_mat;

#define LINC_AT(m, i, j) ((m)->data[(size_t)(i) * (m)->cols + (size_t)(j)])

typedef enum linc_trans { LINC_NO_TRANS = 0, LINC_TRANS = 1 } linc_trans;
typedef enum linc_uplo { LINC_LOWER = 0, LINC_UPPER = 1 } linc_uplo;
typedef enum linc_diag { LINC_NON_UNIT = 0, LINC_UNIT = 1 } linc_diag;

typedef enum linc_norm {
    LINC_NORM_1,   /* max column sum of |a_ij|       */
    LINC_NORM_INF, /* max row sum of |a_ij|          */
    LINC_NORM_FRO, /* sqrt(sum a_ij^2), overflow-safe */
    LINC_NORM_MAX  /* max |a_ij| (not a sub-multiplicative norm) */
} linc_norm;

/* ------------------------------------------------------------------------ */
/* Construction and destruction                                              */
/* ------------------------------------------------------------------------ */

/* Zero-initialised vector of length n. Returns NULL on allocation failure. */
linc_vec *linc_vec_new(size_t n);
/* New vector holding a copy of src[0..n). */
linc_vec *linc_vec_from(const double *src, size_t n);
linc_vec *linc_vec_clone(const linc_vec *v);
/* Release a vector from linc_vec_new/from/clone. NULL is a no-op. */
void linc_vec_free(linc_vec *v);
/* Non-owning view over caller memory. Never pass a view to linc_vec_free. */
linc_vec linc_vec_view(double *data, size_t n);

/* Zero-initialised rows x cols matrix. Returns NULL on failure/overflow. */
linc_mat *linc_mat_new(size_t rows, size_t cols);
/* New matrix from a row-major array of rows * cols doubles. */
linc_mat *linc_mat_from(const double *src, size_t rows, size_t cols);
linc_mat *linc_mat_identity(size_t n);
linc_mat *linc_mat_clone(const linc_mat *m);
void linc_mat_free(linc_mat *m);
linc_mat linc_mat_view(double *data, size_t rows, size_t cols);

/* Copy contents; shapes must match exactly. */
linc_status linc_vec_copy(const linc_vec *src, linc_vec *dst);
linc_status linc_mat_copy(const linc_mat *src, linc_mat *dst);

/* In-place setters. NULL is a no-op. */
void linc_vec_fill(linc_vec *v, double value);
void linc_mat_fill(linc_mat *m, double value);
/* Ones on the main diagonal, zeros elsewhere (works for non-square too). */
void linc_mat_set_identity(linc_mat *m);

/* Row i / column j as a new vector (row can also be viewed directly with
 * linc_vec_view(&LINC_AT(m, i, 0), m->cols)). */
linc_status linc_mat_get_col(const linc_mat *m, size_t j, linc_vec *out);
linc_status linc_mat_set_col(linc_mat *m, size_t j, const linc_vec *v);

/* ------------------------------------------------------------------------ */
/* Level 1: vector-vector, O(n)                                              */
/* ------------------------------------------------------------------------ */

linc_status linc_vec_dot(const linc_vec *x, const linc_vec *y, double *out);
/* y <- alpha * x + y */
linc_status linc_vec_axpy(double alpha, const linc_vec *x, linc_vec *y);
/* x <- alpha * x. NULL is a no-op. */
void linc_vec_scale(linc_vec *x, double alpha);
/* out <- x + y and out <- x - y. out may alias x and/or y. */
linc_status linc_vec_add(const linc_vec *x, const linc_vec *y, linc_vec *out);
linc_status linc_vec_sub(const linc_vec *x, const linc_vec *y, linc_vec *out);

/* Euclidean norm without spurious overflow/underflow (like BLAS dnrm2). */
linc_status linc_vec_norm2(const linc_vec *x, double *out);
linc_status linc_vec_norm1(const linc_vec *x, double *out);
linc_status linc_vec_norm_inf(const linc_vec *x, double *out);

/* ------------------------------------------------------------------------ */
/* Level 2: matrix-vector, O(n^2)                                            */
/* ------------------------------------------------------------------------ */

/* y <- alpha * op(A) * x + beta * y, where op(A) is A or A^T.
 * If beta == 0, y is not read (NaNs in y do not propagate), as in BLAS.
 * y must not alias x. */
linc_status linc_gemv(linc_trans trans, double alpha, const linc_mat *A,
                      const linc_vec *x, double beta, linc_vec *y);

/* A <- A + alpha * x * y^T (rank-1 update). */
linc_status linc_ger(double alpha, const linc_vec *x, const linc_vec *y,
                     linc_mat *A);

/* Solve op(T) * x = b in place (b is overwritten with x), T triangular.
 * Only the uplo triangle of T is read. Returns LINC_ERR_SINGULAR if a
 * diagonal element is exactly zero (and diag == LINC_NON_UNIT). */
linc_status linc_trsv(linc_uplo uplo, linc_trans trans, linc_diag diag,
                      const linc_mat *T, linc_vec *b);

/* ------------------------------------------------------------------------ */
/* Level 3: matrix-matrix, O(n^3)                                            */
/* ------------------------------------------------------------------------ */

/* The GEMM algorithms are exposed individually because comparing them is the
 * point of the benchmarks. They all compute the same thing. */
typedef enum linc_gemm_algo {
    LINC_GEMM_AUTO = 0,  /* pick the best one for the given shape        */
    LINC_GEMM_NAIVE,     /* textbook i-j-k triple loop                   */
    LINC_GEMM_REORDERED, /* i-k-j order: unit stride in the inner loop   */
    LINC_GEMM_BLOCKED,   /* cache tiling on top of i-k-j                 */
    LINC_GEMM_PACKED     /* BLIS-style packing + register micro-kernel   */
} linc_gemm_algo;

/* C <- alpha * A * B + beta * C.
 * If beta == 0, C is not read. C must not alias A or B (LINC_ERR_ALIAS).
 * PACKED (and AUTO) allocate scratch memory; if that fails they fall back to
 * the allocation-free BLOCKED kernel, so GEMM never returns LINC_ERR_ALLOC. */
linc_status linc_gemm_ex(linc_gemm_algo algo, double alpha, const linc_mat *A,
                         const linc_mat *B, double beta, linc_mat *C);
linc_status linc_gemm(double alpha, const linc_mat *A, const linc_mat *B,
                      double beta, linc_mat *C);
/* C <- A * B */
linc_status linc_mat_mul(const linc_mat *A, const linc_mat *B, linc_mat *C);

/* ------------------------------------------------------------------------ */
/* Elementwise and whole-matrix operations                                   */
/* ------------------------------------------------------------------------ */

/* out <- A^T. out must be cols x rows. In place (out == A) only if square. */
linc_status linc_mat_transpose(const linc_mat *A, linc_mat *out);
/* out <- A + B, out <- A - B. out may alias A and/or B. */
linc_status linc_mat_add(const linc_mat *A, const linc_mat *B, linc_mat *out);
linc_status linc_mat_sub(const linc_mat *A, const linc_mat *B, linc_mat *out);
void linc_mat_scale(linc_mat *A, double alpha);
linc_status linc_mat_trace(const linc_mat *A, double *out);
linc_status linc_mat_norm(const linc_mat *A, linc_norm kind, double *out);
/* out <- max |A_ij - B_ij|. Handy for tests and convergence checks. */
linc_status linc_mat_max_abs_diff(const linc_mat *A, const linc_mat *B,
                                  double *out);
/* *out <- 1 if |A_ij - A_ji| <= tol * max|A| for all i, j, else 0. */
linc_status linc_mat_is_symmetric(const linc_mat *A, double tol, int *out);

/* ------------------------------------------------------------------------ */
/* LU decomposition with partial pivoting: P * A = L * U                     */
/* ------------------------------------------------------------------------ */

typedef struct linc_lu {
    linc_mat *lu;  /* n x n; strict lower = L (unit diagonal implied), upper = U */
    size_t *perm;  /* row i of P*A is row perm[i] of A                         */
    int sign;      /* det(P) = +1 or -1                                         */
    int singular;  /* 1 if some |u_kk| <= n * eps * max|A| (see linc_lu_factor) */
} linc_lu;

/* Factor a square matrix. On success the caller owns *out and must release
 * it with linc_lu_free. The factorization always completes; a numerically
 * singular matrix is reported through out->singular (not an error) so that
 * the determinant can still be computed. Solve/inverse then fail with
 * LINC_ERR_SINGULAR. */
linc_status linc_lu_factor(const linc_mat *A, linc_lu *out);
void linc_lu_free(linc_lu *f);
/* Solve A x = b. x may alias b. */
linc_status linc_lu_solve(const linc_lu *f, const linc_vec *b, linc_vec *x);
/* Solve A X = B for many right-hand sides. X may alias B. */
linc_status linc_lu_solve_mat(const linc_lu *f, const linc_mat *B, linc_mat *X);
linc_status linc_lu_det(const linc_lu *f, double *det);
linc_status linc_lu_inverse(const linc_lu *f, linc_mat *inv);

/* ------------------------------------------------------------------------ */
/* Cholesky decomposition: A = L * L^T for symmetric positive definite A     */
/* ------------------------------------------------------------------------ */

/* Only the lower triangle of A is read. L receives the lower factor with the
 * strict upper triangle zeroed. L may alias A (in-place factorization). */
linc_status linc_cholesky(const linc_mat *A, linc_mat *L);
/* Solve A x = b given the Cholesky factor L. x may alias b. */
linc_status linc_cholesky_solve(const linc_mat *L, const linc_vec *b,
                                linc_vec *x);

/* ------------------------------------------------------------------------ */
/* QR decomposition (Householder): A = Q * R, A is m x n with m >= n         */
/* ------------------------------------------------------------------------ */

typedef struct linc_qr {
    linc_mat *qr; /* m x n; upper triangle = R, below = Householder vectors */
    double *tau;  /* n Householder scalars                                  */
} linc_qr;

linc_status linc_qr_factor(const linc_mat *A, linc_qr *out);
void linc_qr_free(linc_qr *f);
/* Thin Q (m x n, orthonormal columns). */
linc_status linc_qr_q(const linc_qr *f, linc_mat *Q);
/* R (n x n, upper triangular). */
linc_status linc_qr_r(const linc_qr *f, linc_mat *R);
/* Least squares: x (length n) minimises ||A x - b||_2, b has length m.
 * Returns LINC_ERR_RANK_DEFICIENT if A is numerically rank deficient. */
linc_status linc_qr_solve(const linc_qr *f, const linc_vec *b, linc_vec *x);

/* ------------------------------------------------------------------------ */
/* Symmetric eigendecomposition (cyclic Jacobi): A = V * diag(w) * V^T       */
/* ------------------------------------------------------------------------ */

/* Eigenvalues are returned in ascending order in w (length n). If V is not
 * NULL it receives the orthonormal eigenvectors as columns (V[:, k] belongs
 * to w[k]). Returns LINC_ERR_NOT_SYMMETRIC if A is not symmetric to within
 * a small relative tolerance. */
linc_status linc_eig_sym(const linc_mat *A, linc_vec *w, linc_mat *V);

/* ------------------------------------------------------------------------ */
/* One-shot convenience wrappers (factor, use, free)                         */
/* ------------------------------------------------------------------------ */

/* Solve the square system A x = b. x may alias b. */
linc_status linc_solve(const linc_mat *A, const linc_vec *b, linc_vec *x);
linc_status linc_det(const linc_mat *A, double *det);
linc_status linc_inverse(const linc_mat *A, linc_mat *inv);
/* Least-squares solution of an overdetermined system (m >= n). */
linc_status linc_lstsq(const linc_mat *A, const linc_vec *b, linc_vec *x);

#ifdef __cplusplus
}
#endif

#endif /* LINC_H */
