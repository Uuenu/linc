/* Test suite for linc. No framework: each test is a function, CHECK records
 * failures and keeps going so one run reports everything that is broken.
 *
 * Decompositions are verified by their defining property (P A = L U,
 * Q^T Q = I, A V = V diag(w), ...) on random matrices of many shapes, which
 * catches far more than a handful of hand-computed examples would. */
#include "linc.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failures;

#define CHECK(cond)                                                          \
    do {                                                                     \
        g_checks++;                                                          \
        if (!(cond)) {                                                       \
            g_failures++;                                                    \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

#define CHECK_ST(expr, want)                                                 \
    do {                                                                     \
        linc_status st_ = (expr);                                            \
        g_checks++;                                                          \
        if (st_ != (want)) {                                                 \
            g_failures++;                                                    \
            fprintf(stderr, "  FAIL %s:%d: %s -> %s, expected %s\n",         \
                    __FILE__, __LINE__, #expr, linc_status_str(st_),         \
                    linc_status_str(want));                                  \
        }                                                                    \
    } while (0)

#define CHECK_OK(expr) CHECK_ST(expr, LINC_OK)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double a_ = (a), b_ = (b);                                           \
        g_checks++;                                                          \
        if (!(fabs(a_ - b_) <= (tol))) {                                     \
            g_failures++;                                                    \
            fprintf(stderr, "  FAIL %s:%d: %s = %.17g, expected %.17g\n",    \
                    __FILE__, __LINE__, #a, a_, b_);                         \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------------ */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------ */

/* Deterministic PRNG (splitmix64) so failures are reproducible. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rnd(void)
{
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0 * 2.0 - 1.0; /* [-1, 1) */
}

static linc_mat *rand_mat(size_t r, size_t c)
{
    linc_mat *m = linc_mat_new(r, c);
    for (size_t i = 0; i < r * c; i++)
        m->data[i] = rnd();
    return m;
}

static linc_vec *rand_vec(size_t n)
{
    linc_vec *v = linc_vec_new(n);
    for (size_t i = 0; i < n; i++)
        v->data[i] = rnd();
    return v;
}

/* Random SPD matrix: M^T M + n I. */
static linc_mat *rand_spd(size_t n)
{
    linc_mat *M = rand_mat(n, n), *Mt = linc_mat_new(n, n),
             *S = linc_mat_new(n, n);
    linc_mat_transpose(M, Mt);
    linc_mat_mul(Mt, M, S);
    for (size_t i = 0; i < n; i++)
        LINC_AT(S, i, i) += (double)n;
    linc_mat_free(M);
    linc_mat_free(Mt);
    return S;
}

static double max_diff(const linc_mat *A, const linc_mat *B)
{
    double d = INFINITY;
    linc_mat_max_abs_diff(A, B, &d);
    return d;
}

/* Reference product, deliberately as simple as possible. */
static void ref_mul(const linc_mat *A, const linc_mat *B, linc_mat *C)
{
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < B->cols; j++) {
            long double s = 0;
            for (size_t k = 0; k < A->cols; k++)
                s += (long double)LINC_AT(A, i, k) * LINC_AT(B, k, j);
            LINC_AT(C, i, j) = (double)s;
        }
}

/* ------------------------------------------------------------------------ */
/* Construction                                                              */
/* ------------------------------------------------------------------------ */

static void test_construction(void)
{
    linc_mat *m = linc_mat_new(3, 4);
    CHECK(m && m->rows == 3 && m->cols == 4);
    CHECK(((uintptr_t)m->data & 63) == 0); /* documented 64-byte alignment */
    for (size_t i = 0; i < 12; i++)
        CHECK(m->data[i] == 0.0);
    linc_mat_free(m);

    /* Overflowing sizes must fail cleanly, not wrap around. */
    CHECK(linc_mat_new(SIZE_MAX / 2, 4) == NULL);
    CHECK(linc_vec_new(SIZE_MAX / 4) == NULL);

    /* Empty objects are legal. */
    linc_mat *e = linc_mat_new(0, 5);
    CHECK(e != NULL);
    linc_mat_free(e);
    linc_mat_free(NULL);
    linc_vec_free(NULL);

    double src[] = {1, 2, 3, 4, 5, 6};
    linc_mat *f = linc_mat_from(src, 2, 3);
    CHECK(LINC_AT(f, 1, 0) == 4 && LINC_AT(f, 0, 2) == 3);
    linc_mat *c = linc_mat_clone(f);
    CHECK(c->data != f->data && max_diff(c, f) == 0.0);

    linc_mat *I = linc_mat_identity(3);
    double tr;
    CHECK_OK(linc_mat_trace(I, &tr));
    CHECK(tr == 3.0);

    linc_vec *col = linc_vec_new(2);
    CHECK_OK(linc_mat_get_col(f, 1, col));
    CHECK(col->data[0] == 2 && col->data[1] == 5);
    CHECK_ST(linc_mat_get_col(f, 3, col), LINC_ERR_ARG);
    col->data[0] = 9;
    CHECK_OK(linc_mat_set_col(f, 2, col));
    CHECK(LINC_AT(f, 0, 2) == 9);

    double buf[4] = {0};
    linc_mat v = linc_mat_view(buf, 2, 2);
    linc_mat_set_identity(&v);
    CHECK(buf[0] == 1 && buf[1] == 0 && buf[3] == 1);

    CHECK_ST(linc_mat_copy(f, I), LINC_ERR_DIM);
    CHECK(strcmp(linc_status_str(LINC_ERR_SINGULAR), "matrix is singular") == 0);
    CHECK(linc_status_str((linc_status)999) != NULL);

    linc_vec_free(col);
    linc_mat_free(I);
    linc_mat_free(c);
    linc_mat_free(f);
}

/* ------------------------------------------------------------------------ */
/* Level 1                                                                   */
/* ------------------------------------------------------------------------ */

static void test_level1(void)
{
    double xs[] = {1, 2, 3, 4, 5}, ys[] = {5, 4, 3, 2, 1};
    linc_vec x = linc_vec_view(xs, 5), y = linc_vec_view(ys, 5);
    double d;
    CHECK_OK(linc_vec_dot(&x, &y, &d));
    CHECK(d == 35.0);

    linc_vec *z = linc_vec_new(5);
    CHECK_OK(linc_vec_add(&x, &y, z));
    CHECK(z->data[0] == 6 && z->data[4] == 6);
    CHECK_OK(linc_vec_sub(&x, &y, z));
    CHECK(z->data[0] == -4 && z->data[4] == 4);
    CHECK_OK(linc_vec_axpy(2.0, &x, z)); /* z = (x - y) + 2x */
    CHECK(z->data[0] == -2 && z->data[4] == 14);
    linc_vec_scale(z, 0.5);
    CHECK(z->data[4] == 7);

    CHECK_OK(linc_vec_norm1(&x, &d));
    CHECK(d == 15);
    CHECK_OK(linc_vec_norm_inf(&y, &d));
    CHECK(d == 5);
    CHECK_OK(linc_vec_norm2(&x, &d));
    CHECK_NEAR(d, sqrt(55.0), 1e-15);

    /* norm2 must survive values whose squares over/underflow. */
    double big[] = {3e200, 4e200}, tiny[] = {3e-200, 4e-200}, zero[] = {0, 0};
    linc_vec b = linc_vec_view(big, 2), t = linc_vec_view(tiny, 2),
             zv = linc_vec_view(zero, 2);
    CHECK_OK(linc_vec_norm2(&b, &d));
    CHECK_NEAR(d / 5e200, 1.0, 1e-15);
    CHECK_OK(linc_vec_norm2(&t, &d));
    CHECK_NEAR(d / 5e-200, 1.0, 1e-15);
    CHECK_OK(linc_vec_norm2(&zv, &d));
    CHECK(d == 0.0);
    double withinf[] = {1, INFINITY};
    linc_vec wi = linc_vec_view(withinf, 2);
    CHECK_OK(linc_vec_norm2(&wi, &d));
    CHECK(isinf(d));

    linc_vec *short_v = linc_vec_new(3);
    CHECK_ST(linc_vec_dot(&x, short_v, &d), LINC_ERR_DIM);
    CHECK_ST(linc_vec_dot(NULL, &y, &d), LINC_ERR_NULL);
    linc_vec_free(short_v);
    linc_vec_free(z);
}

/* ------------------------------------------------------------------------ */
/* Level 2                                                                   */
/* ------------------------------------------------------------------------ */

static void test_level2(void)
{
    double as[] = {1, 2, 3, 4, 5, 6}; /* 2 x 3 */
    linc_mat A = linc_mat_view(as, 2, 3);
    double xs[] = {1, 1, 1}, ys[] = {NAN, NAN};
    linc_vec x = linc_vec_view(xs, 3), y = linc_vec_view(ys, 2);

    /* beta = 0 must ignore NaN garbage in y. */
    CHECK_OK(linc_gemv(LINC_NO_TRANS, 1.0, &A, &x, 0.0, &y));
    CHECK(ys[0] == 6 && ys[1] == 15);
    CHECK_OK(linc_gemv(LINC_NO_TRANS, 2.0, &A, &x, 1.0, &y));
    CHECK(ys[0] == 18 && ys[1] == 45);

    double ts[] = {1, 2, 0};
    linc_vec t = linc_vec_view(ts, 3);
    CHECK_OK(linc_gemv(LINC_TRANS, 1.0, &A, &y, 0.0, &t)); /* A^T y */
    CHECK(ts[0] == 18 + 4 * 45 && ts[2] == 3 * 18 + 6 * 45);

    CHECK_ST(linc_gemv(LINC_NO_TRANS, 1.0, &A, &y, 0.0, &x), LINC_ERR_DIM);
    linc_vec xalias = linc_vec_view(as, 3), yalias = linc_vec_view(as + 1, 2);
    CHECK_ST(linc_gemv(LINC_NO_TRANS, 1.0, &A, &xalias, 0.0, &yalias),
             LINC_ERR_ALIAS);

    /* ger: A += x y^T */
    linc_mat *G = linc_mat_new(2, 3);
    double us[] = {1, 2}, vs[] = {3, 4, 5};
    linc_vec u = linc_vec_view(us, 2), v = linc_vec_view(vs, 3);
    CHECK_OK(linc_ger(2.0, &u, &v, G));
    CHECK(LINC_AT(G, 1, 2) == 20 && LINC_AT(G, 0, 0) == 6);
    linc_mat_free(G);

    /* trsv: all four (uplo, trans) combinations against a gemv round trip. */
    for (size_t n = 1; n <= 40; n += 13) {
        linc_mat *T = rand_mat(n, n);
        for (size_t i = 0; i < n; i++)
            LINC_AT(T, i, i) += 4.0 * (rnd() < 0 ? -1 : 1); /* well conditioned */
        for (int uplo = 0; uplo < 2; uplo++)
            for (int tr = 0; tr < 2; tr++)
                for (int unit = 0; unit < 2; unit++) {
                    /* Build the explicit triangle to multiply with. */
                    linc_mat *Tri = linc_mat_new(n, n);
                    for (size_t i = 0; i < n; i++)
                        for (size_t j = 0; j < n; j++) {
                            int in = uplo == LINC_LOWER ? j <= i : j >= i;
                            double val = in ? LINC_AT(T, i, j) : 0.0;
                            if (i == j && unit)
                                val = 1.0;
                            LINC_AT(Tri, i, j) = val;
                        }
                    linc_vec *xt = rand_vec(n), *b = linc_vec_new(n);
                    linc_gemv((linc_trans)tr, 1.0, Tri, xt, 0.0, b);
                    CHECK_OK(linc_trsv((linc_uplo)uplo, (linc_trans)tr,
                                       (linc_diag)unit, T, b));
                    for (size_t i = 0; i < n; i++)
                        CHECK_NEAR(b->data[i], xt->data[i], 1e-12);
                    linc_vec_free(xt);
                    linc_vec_free(b);
                    linc_mat_free(Tri);
                }
        linc_mat_free(T);
    }

    double zs[] = {1, 0, 0, 0};
    linc_mat Z = linc_mat_view(zs, 2, 2);
    double bs[] = {1, 1};
    linc_vec bv = linc_vec_view(bs, 2);
    CHECK_ST(linc_trsv(LINC_UPPER, LINC_NO_TRANS, LINC_NON_UNIT, &Z, &bv),
             LINC_ERR_SINGULAR);
    CHECK(bs[0] == 1 && bs[1] == 1); /* untouched on failure */
}

/* ------------------------------------------------------------------------ */
/* Level 3                                                                   */
/* ------------------------------------------------------------------------ */

static void test_gemm(void)
{
    /* Shapes chosen to hit every edge: 1s, non-multiples of the 4 x 8
     * micro-tile, and sizes crossing the KC = 256 / MC = 128 block edges. */
    static const size_t shapes[][3] = {
        {1, 1, 1},   {1, 7, 1},     {7, 1, 9},    {3, 5, 7},
        {4, 8, 4},   {5, 9, 3},     {17, 33, 65}, {64, 64, 64},
        {65, 63, 67}, {130, 257, 129}, {200, 100, 300}, {0, 4, 3},
        {4, 0, 3},   {4, 3, 0},
    };
    const linc_gemm_algo algos[] = {LINC_GEMM_AUTO, LINC_GEMM_NAIVE,
                                    LINC_GEMM_REORDERED, LINC_GEMM_BLOCKED,
                                    LINC_GEMM_PACKED};

    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t m = shapes[s][0], k = shapes[s][1], n = shapes[s][2];
        linc_mat *A = rand_mat(m, k), *B = rand_mat(k, n);
        linc_mat *C0 = rand_mat(m, n), *ref = linc_mat_new(m, n);
        double alpha = 1.5, beta = -0.5;

        /* ref = alpha A B + beta C0 */
        ref_mul(A, B, ref);
        for (size_t i = 0; i < m * n; i++)
            ref->data[i] = alpha * ref->data[i] + beta * C0->data[i];

        for (size_t a = 0; a < sizeof algos / sizeof algos[0]; a++) {
            linc_mat *C = linc_mat_clone(C0);
            CHECK_OK(linc_gemm_ex(algos[a], alpha, A, B, beta, C));
            double tol = 1e-13 * (double)(k + 1);
            if (!(max_diff(C, ref) <= tol))
                fprintf(stderr, "  gemm algo %d shape %zux%zux%zu\n",
                        (int)algos[a], m, k, n);
            CHECK(max_diff(C, ref) <= tol);

            /* beta == 0 must overwrite NaNs, not propagate them. */
            linc_mat_fill(C, NAN);
            CHECK_OK(linc_gemm_ex(algos[a], 1.0, A, B, 0.0, C));
            linc_mat *AB = linc_mat_new(m, n);
            ref_mul(A, B, AB);
            CHECK(max_diff(C, AB) <= tol);
            linc_mat_free(AB);
            linc_mat_free(C);
        }
        linc_mat_free(A);
        linc_mat_free(B);
        linc_mat_free(C0);
        linc_mat_free(ref);
    }

    linc_mat *A = rand_mat(4, 4), *B = rand_mat(4, 3), *C = linc_mat_new(3, 3);
    CHECK_ST(linc_mat_mul(A, B, C), LINC_ERR_DIM);
    CHECK_ST(linc_mat_mul(A, A, A), LINC_ERR_ALIAS);
    CHECK_ST(linc_gemm_ex((linc_gemm_algo)42, 1, A, A, 0, C), LINC_ERR_DIM);
    linc_mat *D = linc_mat_new(4, 4);
    CHECK_ST(linc_gemm_ex((linc_gemm_algo)42, 1, A, A, 0, D), LINC_ERR_ARG);
    linc_mat_free(D);
    linc_mat_free(A);
    linc_mat_free(B);
    linc_mat_free(C);
}

static void test_matrix_ops(void)
{
    linc_mat *A = rand_mat(37, 53), *At = linc_mat_new(53, 37);
    CHECK_OK(linc_mat_transpose(A, At));
    int ok = 1;
    for (size_t i = 0; i < 37; i++)
        for (size_t j = 0; j < 53; j++)
            ok &= LINC_AT(A, i, j) == LINC_AT(At, j, i);
    CHECK(ok);
    CHECK_ST(linc_mat_transpose(A, A), LINC_ERR_DIM);

    linc_mat *S = rand_mat(5, 5), *S2 = linc_mat_clone(S);
    CHECK_OK(linc_mat_transpose(S, S)); /* square, in place */
    CHECK(LINC_AT(S, 1, 3) == LINC_AT(S2, 3, 1));

    double as[] = {1, -2, 3, -4};
    linc_mat M = linc_mat_view(as, 2, 2);
    double n;
    CHECK_OK(linc_mat_norm(&M, LINC_NORM_1, &n));
    CHECK(n == 6);
    CHECK_OK(linc_mat_norm(&M, LINC_NORM_INF, &n));
    CHECK(n == 7);
    CHECK_OK(linc_mat_norm(&M, LINC_NORM_MAX, &n));
    CHECK(n == 4);
    CHECK_OK(linc_mat_norm(&M, LINC_NORM_FRO, &n));
    CHECK_NEAR(n, sqrt(30.0), 1e-15);
    CHECK_ST(linc_mat_norm(&M, (linc_norm)99, &n), LINC_ERR_ARG);

    CHECK_OK(linc_mat_add(S, S2, S)); /* aliasing output allowed */
    CHECK_OK(linc_mat_sub(S, S2, S));
    linc_mat_scale(S, 2.0);

    int sym;
    linc_mat *P = rand_spd(6);
    CHECK_OK(linc_mat_is_symmetric(P, 1e-12, &sym));
    CHECK(sym == 1);
    LINC_AT(P, 0, 5) += 1e-3;
    CHECK_OK(linc_mat_is_symmetric(P, 1e-12, &sym));
    CHECK(sym == 0);

    linc_mat_free(P);
    linc_mat_free(A);
    linc_mat_free(At);
    linc_mat_free(S);
    linc_mat_free(S2);
}

/* ------------------------------------------------------------------------ */
/* Decompositions                                                            */
/* ------------------------------------------------------------------------ */

static void test_lu(void)
{
    for (size_t n = 1; n <= 80; n += 7) {
        linc_mat *A = rand_mat(n, n);
        linc_lu f;
        CHECK_OK(linc_lu_factor(A, &f));
        CHECK(!f.singular);

        /* Rebuild L and U and check L U == P A. */
        linc_mat *L = linc_mat_new(n, n), *U = linc_mat_new(n, n),
                 *LU = linc_mat_new(n, n), *PA = linc_mat_new(n, n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                double v = LINC_AT(f.lu, i, j);
                LINC_AT(L, i, j) = j < i ? v : (i == j ? 1.0 : 0.0);
                LINC_AT(U, i, j) = j >= i ? v : 0.0;
                LINC_AT(PA, i, j) = LINC_AT(A, f.perm[i], j);
                if (j < i)
                    CHECK(fabs(v) <= 1.0); /* partial pivoting bound */
            }
        linc_mat_mul(L, U, LU);
        CHECK(max_diff(LU, PA) <= 1e-13 * (double)n);

        /* Solve, check the residual ||A x - b||. */
        linc_vec *xt = rand_vec(n), *b = linc_vec_new(n), *x = linc_vec_new(n);
        linc_gemv(LINC_NO_TRANS, 1.0, A, xt, 0.0, b);
        CHECK_OK(linc_lu_solve(&f, b, x));
        for (size_t i = 0; i < n; i++)
            CHECK_NEAR(x->data[i], xt->data[i], 1e-9);
        CHECK_OK(linc_lu_solve(&f, b, b)); /* in place */
        for (size_t i = 0; i < n; i++)
            CHECK_NEAR(b->data[i], x->data[i], 1e-15);

        /* Inverse: A * inv(A) == I. */
        linc_mat *inv = linc_mat_new(n, n), *I = linc_mat_new(n, n),
                 *Id = linc_mat_identity(n);
        CHECK_OK(linc_lu_inverse(&f, inv));
        linc_mat_mul(A, inv, I);
        CHECK(max_diff(I, Id) <= 1e-10);

        linc_mat_free(inv);
        linc_mat_free(I);
        linc_mat_free(Id);
        linc_vec_free(xt);
        linc_vec_free(b);
        linc_vec_free(x);
        linc_mat_free(L);
        linc_mat_free(U);
        linc_mat_free(LU);
        linc_mat_free(PA);
        linc_lu_free(&f);
        linc_mat_free(A);
    }

    /* Determinants with known values. */
    double as[] = {2, 0, 1, 1, 3, 2, 1, 1, 2}; /* det = 6 */
    linc_mat A = linc_mat_view(as, 3, 3);
    double det;
    CHECK_OK(linc_det(&A, &det));
    CHECK_NEAR(det, 6.0, 1e-14);
    double ps[] = {0, 1, 1, 0}; /* a permutation: det = -1 */
    linc_mat P = linc_mat_view(ps, 2, 2);
    CHECK_OK(linc_det(&P, &det));
    CHECK(det == -1.0);
    linc_mat E = linc_mat_view(NULL, 0, 0);
    CHECK_OK(linc_det(&E, &det));
    CHECK(det == 1.0); /* empty product */

    /* Pivoting matters: this fails badly without it. */
    double ts[] = {1e-20, 1, 1, 1}, tb[] = {1, 2}, tx[2];
    linc_mat T = linc_mat_view(ts, 2, 2);
    linc_vec b = linc_vec_view(tb, 2), x = linc_vec_view(tx, 2);
    CHECK_OK(linc_solve(&T, &b, &x));
    CHECK_NEAR(tx[0], 1.0, 1e-15);
    CHECK_NEAR(tx[1], 1.0, 1e-15);

    /* Singular: exactly and numerically. */
    double ss[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}, sb[] = {1, 2, 3}, sx[3];
    linc_mat S = linc_mat_view(ss, 3, 3);
    linc_vec sbv = linc_vec_view(sb, 3), sxv = linc_vec_view(sx, 3);
    CHECK_ST(linc_solve(&S, &sbv, &sxv), LINC_ERR_SINGULAR);
    linc_lu f;
    CHECK_OK(linc_lu_factor(&S, &f));
    CHECK(f.singular == 1);
    CHECK_OK(linc_lu_det(&f, &det));
    CHECK(fabs(det) < 1e-12);
    linc_lu_free(&f);
    linc_lu_free(&f); /* double free is safe: pointers were cleared */

    linc_mat *Z = linc_mat_new(3, 3), *Zi = linc_mat_new(3, 3);
    CHECK_ST(linc_inverse(Z, Zi), LINC_ERR_SINGULAR);
    CHECK_OK(linc_det(Z, &det));
    CHECK(det == 0.0);
    linc_mat_free(Z);
    linc_mat_free(Zi);

    linc_mat *R = rand_mat(3, 4);
    CHECK_ST(linc_lu_factor(R, &f), LINC_ERR_DIM);
    linc_mat_free(R);

    /* Multiple right-hand sides. */
    linc_mat *M = rand_mat(20, 20), *Xt = rand_mat(20, 5),
             *B = linc_mat_new(20, 5), *X = linc_mat_new(20, 5);
    linc_mat_mul(M, Xt, B);
    CHECK_OK(linc_lu_factor(M, &f));
    CHECK_OK(linc_lu_solve_mat(&f, B, X));
    CHECK(max_diff(X, Xt) <= 1e-10);
    CHECK_OK(linc_lu_solve_mat(&f, B, B));
    CHECK(max_diff(B, Xt) <= 1e-10);
    linc_lu_free(&f);
    linc_mat_free(M);
    linc_mat_free(Xt);
    linc_mat_free(B);
    linc_mat_free(X);
}

static void test_cholesky(void)
{
    for (size_t n = 1; n <= 60; n += 9) {
        linc_mat *A = rand_spd(n), *L = linc_mat_new(n, n),
                 *Lt = linc_mat_new(n, n), *LLt = linc_mat_new(n, n);
        CHECK_OK(linc_cholesky(A, L));
        for (size_t i = 0; i < n; i++) {
            CHECK(LINC_AT(L, i, i) > 0);
            for (size_t j = i + 1; j < n; j++)
                CHECK(LINC_AT(L, i, j) == 0.0);
        }
        linc_mat_transpose(L, Lt);
        linc_mat_mul(L, Lt, LLt);
        CHECK(max_diff(LLt, A) <= 1e-12 * (double)n);

        linc_vec *xt = rand_vec(n), *b = linc_vec_new(n), *x = linc_vec_new(n);
        linc_gemv(LINC_NO_TRANS, 1.0, A, xt, 0.0, b);
        CHECK_OK(linc_cholesky_solve(L, b, x));
        for (size_t i = 0; i < n; i++)
            CHECK_NEAR(x->data[i], xt->data[i], 1e-12);

        /* In-place factorization matches. */
        linc_mat *A2 = linc_mat_clone(A);
        CHECK_OK(linc_cholesky(A2, A2));
        CHECK(max_diff(A2, L) == 0.0);

        linc_mat_free(A2);
        linc_vec_free(xt);
        linc_vec_free(b);
        linc_vec_free(x);
        linc_mat_free(A);
        linc_mat_free(L);
        linc_mat_free(Lt);
        linc_mat_free(LLt);
    }

    double ns[] = {1, 2, 2, 1}; /* symmetric, eigenvalues 3 and -1 */
    linc_mat N = linc_mat_view(ns, 2, 2);
    linc_mat *L = linc_mat_new(2, 2);
    CHECK_ST(linc_cholesky(&N, L), LINC_ERR_NOT_SPD);
    double nan_s[] = {NAN, 0, 0, 1};
    linc_mat Nn = linc_mat_view(nan_s, 2, 2);
    CHECK_ST(linc_cholesky(&Nn, L), LINC_ERR_NOT_SPD);
    linc_mat_free(L);
}

static void test_qr(void)
{
    static const size_t shapes[][2] = {{1, 1}, {5, 3}, {10, 10}, {40, 7},
                                       {64, 33}, {100, 1}, {3, 0}};
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t m = shapes[s][0], n = shapes[s][1];
        linc_mat *A = rand_mat(m, n);
        linc_qr f;
        CHECK_OK(linc_qr_factor(A, &f));
        linc_mat *Q = linc_mat_new(m, n), *R = linc_mat_new(n, n),
                 *QR = linc_mat_new(m, n), *Qt = linc_mat_new(n, m),
                 *QtQ = linc_mat_new(n, n), *I = linc_mat_identity(n);
        CHECK_OK(linc_qr_q(&f, Q));
        CHECK_OK(linc_qr_r(&f, R));
        linc_mat_mul(Q, R, QR);
        CHECK(max_diff(QR, A) <= 1e-13 * (double)(m + 1));
        linc_mat_transpose(Q, Qt);
        linc_mat_mul(Qt, Q, QtQ);
        CHECK(max_diff(QtQ, I) <= 1e-13 * (double)(m + 1));
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < i; j++)
                CHECK(LINC_AT(R, i, j) == 0.0);

        linc_mat_free(Q);
        linc_mat_free(R);
        linc_mat_free(QR);
        linc_mat_free(Qt);
        linc_mat_free(QtQ);
        linc_mat_free(I);
        linc_qr_free(&f);
        linc_mat_free(A);
    }

    /* Least squares: fit y = 2 + 3t exactly, then check the normal
     * equations A^T (A x - b) = 0 on noisy random data. */
    double ts[] = {0, 1, 2, 3, 4};
    linc_mat *A = linc_mat_new(5, 2);
    linc_vec *b = linc_vec_new(5), *x = linc_vec_new(2);
    for (size_t i = 0; i < 5; i++) {
        LINC_AT(A, i, 0) = 1.0;
        LINC_AT(A, i, 1) = ts[i];
        b->data[i] = 2.0 + 3.0 * ts[i];
    }
    CHECK_OK(linc_lstsq(A, b, x));
    CHECK_NEAR(x->data[0], 2.0, 1e-13);
    CHECK_NEAR(x->data[1], 3.0, 1e-13);
    linc_mat_free(A);
    linc_vec_free(b);
    linc_vec_free(x);

    A = rand_mat(50, 6);
    b = rand_vec(50);
    x = linc_vec_new(6);
    linc_vec *r = linc_vec_new(50), *g = linc_vec_new(6);
    CHECK_OK(linc_lstsq(A, b, x));
    linc_vec_copy(b, r);
    linc_gemv(LINC_NO_TRANS, 1.0, A, x, -1.0, r); /* r = A x - b */
    linc_gemv(LINC_TRANS, 1.0, A, r, 0.0, g);      /* g = A^T r     */
    double gn;
    linc_vec_norm_inf(g, &gn);
    CHECK(gn <= 1e-12);

    /* Duplicate column: rank deficient. */
    for (size_t i = 0; i < 50; i++)
        LINC_AT(A, i, 3) = LINC_AT(A, i, 1);
    CHECK_ST(linc_lstsq(A, b, x), LINC_ERR_RANK_DEFICIENT);

    linc_mat *W = rand_mat(3, 5);
    linc_qr f;
    CHECK_ST(linc_qr_factor(W, &f), LINC_ERR_DIM); /* m < n */
    linc_mat_free(W);
    linc_mat_free(A);
    linc_vec_free(b);
    linc_vec_free(x);
    linc_vec_free(r);
    linc_vec_free(g);
}

static void test_eig(void)
{
    /* Known spectrum: [[2,1],[1,2]] has eigenvalues 1 and 3. */
    double as[] = {2, 1, 1, 2};
    linc_mat A = linc_mat_view(as, 2, 2);
    linc_vec *w = linc_vec_new(2);
    linc_mat *V = linc_mat_new(2, 2);
    CHECK_OK(linc_eig_sym(&A, w, V));
    CHECK_NEAR(w->data[0], 1.0, 1e-15);
    CHECK_NEAR(w->data[1], 3.0, 1e-15);
    CHECK_NEAR(fabs(LINC_AT(V, 0, 0)), sqrt(0.5), 1e-15);
    linc_vec_free(w);
    linc_mat_free(V);

    for (size_t n = 1; n <= 50; n += 7) {
        linc_mat *S = rand_mat(n, n), *St = linc_mat_new(n, n);
        linc_mat_transpose(S, St);
        linc_mat_add(S, St, S); /* symmetrise, indefinite */
        w = linc_vec_new(n);
        V = linc_mat_new(n, n);
        CHECK_OK(linc_eig_sym(S, w, V));

        for (size_t i = 0; i + 1 < n; i++)
            CHECK(w->data[i] <= w->data[i + 1]);

        /* A V == V diag(w) and V^T V == I. */
        linc_mat *AV = linc_mat_new(n, n), *VD = linc_mat_clone(V),
                 *Vt = linc_mat_new(n, n), *VtV = linc_mat_new(n, n),
                 *I = linc_mat_identity(n);
        linc_mat_mul(S, V, AV);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                LINC_AT(VD, i, j) *= w->data[j];
        CHECK(max_diff(AV, VD) <= 1e-12 * (double)n);
        linc_mat_transpose(V, Vt);
        linc_mat_mul(Vt, V, VtV);
        CHECK(max_diff(VtV, I) <= 1e-12 * (double)n);

        /* Trace equals the sum of eigenvalues. */
        double tr, sum = 0;
        linc_mat_trace(S, &tr);
        for (size_t i = 0; i < n; i++)
            sum += w->data[i];
        CHECK_NEAR(sum, tr, 1e-11 * (double)n);

        /* Eigenvalues only (V == NULL) gives the same values. */
        linc_vec *w2 = linc_vec_new(n);
        CHECK_OK(linc_eig_sym(S, w2, NULL));
        for (size_t i = 0; i < n; i++)
            CHECK_NEAR(w2->data[i], w->data[i], 1e-12 * (double)n);

        linc_vec_free(w2);
        linc_mat_free(AV);
        linc_mat_free(VD);
        linc_mat_free(Vt);
        linc_mat_free(VtV);
        linc_mat_free(I);
        linc_mat_free(S);
        linc_mat_free(St);
        linc_vec_free(w);
        linc_mat_free(V);
    }

    double ns[] = {1, 2, 3, 4};
    linc_mat N = linc_mat_view(ns, 2, 2);
    w = linc_vec_new(2);
    CHECK_ST(linc_eig_sym(&N, w, NULL), LINC_ERR_NOT_SYMMETRIC);
    linc_vec_free(w);
}

int main(void)
{
    struct {
        const char *name;
        void (*fn)(void);
    } tests[] = {
        {"construction", test_construction},
        {"level1", test_level1},
        {"level2", test_level2},
        {"gemm", test_gemm},
        {"matrix_ops", test_matrix_ops},
        {"lu", test_lu},
        {"cholesky", test_cholesky},
        {"qr", test_qr},
        {"eig", test_eig},
    };
    printf("linc %s test suite\n", LINC_VERSION);
    for (size_t i = 0; i < sizeof tests / sizeof tests[0]; i++) {
        int before = g_failures;
        tests[i].fn();
        printf("  %-14s %s\n", tests[i].name,
               g_failures == before ? "ok" : "FAILED");
    }
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
