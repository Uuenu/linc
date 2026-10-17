/* LU decomposition with partial pivoting, and the solve/det/inverse
 * convenience wrappers built on it.
 *
 * Gaussian elimination turns A into an upper triangular U by subtracting
 * multiples of the pivot row from the rows below it. Recording each
 * multiplier l_ik = a_ik / a_kk in the slot it zeroes out gives a unit lower
 * triangular L with A = L * U, stored in the same n x n array.
 *
 * Partial pivoting swaps the row with the largest |a_ik| into the pivot
 * position first, which keeps every |l_ik| <= 1. Without it, a tiny pivot
 * produces huge multipliers and the rounding errors swamp the answer (try
 * [[1e-20, 1], [1, 1]] without pivoting). The swaps are recorded in a
 * permutation, so what we actually factor is P * A = L * U.
 *
 * Cost: 2n^3/3 flops to factor, then 2n^2 per right-hand side to solve.
 */
#include "linc_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

linc_status linc_lu_factor(const linc_mat *A, linc_lu *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    if (A->rows != A->cols)
        return LINC_ERR_DIM;
    size_t n = A->rows;

    linc_mat *lu = linc_mat_clone(A);
    size_t *perm = malloc((n ? n : 1) * sizeof(size_t));
    if (!lu || !perm) {
        linc_mat_free(lu);
        free(perm);
        return LINC_ERR_ALLOC;
    }
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    /* A pivot this small relative to the entries of A is indistinguishable
     * from zero after rounding; the same criterion numpy uses for rank. */
    double tol = (double)n * DBL_EPSILON * linc__max_abs(A->data, n * n);
    int sign = 1, singular = 0;

    for (size_t k = 0; k < n; k++) {
        /* 1. Find the pivot: largest magnitude in column k, rows k..n-1. */
        size_t p = k;
        double best = fabs(LINC_AT(lu, k, k));
        for (size_t i = k + 1; i < n; i++) {
            double v = fabs(LINC_AT(lu, i, k));
            if (v > best) {
                best = v;
                p = i;
            }
        }

        /* 2. Swap it into place (whole rows, so earlier multipliers move
         *    too, keeping L consistent with the final permutation). */
        if (p != k) {
            double *rk = &LINC_AT(lu, k, 0), *rp = &LINC_AT(lu, p, 0);
            for (size_t j = 0; j < n; j++) {
                double t = rk[j];
                rk[j] = rp[j];
                rp[j] = t;
            }
            size_t t = perm[k];
            perm[k] = perm[p];
            perm[p] = t;
            sign = -sign;
        }

        double pivot = LINC_AT(lu, k, k);
        if (!(fabs(pivot) > tol))
            singular = 1;
        if (pivot == 0.0)
            continue; /* column is already zero below the diagonal */

        /* 3. Eliminate below the pivot. Updating row by row (k-i-j order)
         *    makes the innermost loop contiguous in row-major storage. */
        const double *rk = &LINC_AT(lu, k, 0);
        for (size_t i = k + 1; i < n; i++) {
            double *ri = &LINC_AT(lu, i, 0);
            double l = ri[k] / pivot;
            ri[k] = l;
            if (l == 0.0)
                continue;
            for (size_t j = k + 1; j < n; j++)
                ri[j] -= l * rk[j];
        }
    }

    out->lu = lu;
    out->perm = perm;
    out->sign = sign;
    out->singular = singular;
    return LINC_OK;
}

void linc_lu_free(linc_lu *f)
{
    if (!f)
        return;
    linc_mat_free(f->lu);
    free(f->perm);
    f->lu = NULL;
    f->perm = NULL;
}

/* Solve L U X = X in place (X already holds P * B). Operating on whole rows
 * of X handles every right-hand side in one contiguous sweep. */
static void lu_substitute(const linc_mat *lu, linc_mat *X)
{
    size_t n = lu->rows, r = X->cols;
    /* Forward: L y = b, L unit lower triangular. */
    for (size_t i = 0; i < n; i++) {
        double *xi = &LINC_AT(X, i, 0);
        for (size_t k = 0; k < i; k++) {
            double l = LINC_AT(lu, i, k);
            if (l == 0.0)
                continue;
            const double *xk = &LINC_AT(X, k, 0);
            for (size_t j = 0; j < r; j++)
                xi[j] -= l * xk[j];
        }
    }
    /* Backward: U x = y. */
    for (size_t i = n; i-- > 0;) {
        double *xi = &LINC_AT(X, i, 0);
        for (size_t k = i + 1; k < n; k++) {
            double u = LINC_AT(lu, i, k);
            if (u == 0.0)
                continue;
            const double *xk = &LINC_AT(X, k, 0);
            for (size_t j = 0; j < r; j++)
                xi[j] -= u * xk[j];
        }
        double d = LINC_AT(lu, i, i);
        for (size_t j = 0; j < r; j++)
            xi[j] /= d;
    }
}

static linc_status check_factor(const linc_lu *f)
{
    if (!f || !f->lu || !f->perm)
        return LINC_ERR_NULL;
    return LINC_OK;
}

linc_status linc_lu_solve_mat(const linc_lu *f, const linc_mat *B, linc_mat *X)
{
    linc_status st = check_factor(f);
    if (st != LINC_OK)
        return st;
    if (!B || !X)
        return LINC_ERR_NULL;
    size_t n = f->lu->rows;
    if (B->rows != n || X->rows != n || X->cols != B->cols)
        return LINC_ERR_DIM;
    if (linc_mats_overlap(X, f->lu))
        return LINC_ERR_ALIAS;
    if (X->data != B->data && linc_mats_overlap(X, B))
        return LINC_ERR_ALIAS;
    if (f->singular)
        return LINC_ERR_SINGULAR;

    size_t r = B->cols, rowbytes = r * sizeof(double);
    if (n == 0 || r == 0)
        return LINC_OK;

    /* X <- P * B. When solving in place, permute through a copy of B. */
    const double *src = B->data;
    double *tmp = NULL;
    if (X->data == B->data) {
        tmp = malloc(n * rowbytes);
        if (!tmp)
            return LINC_ERR_ALLOC;
        memcpy(tmp, B->data, n * rowbytes);
        src = tmp;
    }
    for (size_t i = 0; i < n; i++)
        memcpy(&LINC_AT(X, i, 0), src + f->perm[i] * r, rowbytes);
    free(tmp);

    lu_substitute(f->lu, X);
    return LINC_OK;
}

linc_status linc_lu_solve(const linc_lu *f, const linc_vec *b, linc_vec *x)
{
    if (!b || !x)
        return LINC_ERR_NULL;
    /* A vector is an n x 1 matrix; reuse the multi-RHS path. The cast is
     * safe because B is only read. */
    linc_mat B = linc_mat_view((double *)b->data, b->n, 1);
    linc_mat X = linc_mat_view(x->data, x->n, 1);
    return linc_lu_solve_mat(f, &B, &X);
}

linc_status linc_lu_det(const linc_lu *f, double *det)
{
    linc_status st = check_factor(f);
    if (st != LINC_OK)
        return st;
    if (!det)
        return LINC_ERR_NULL;
    /* det(P) det(A) = det(L) det(U) = 1 * prod u_kk */
    double d = (double)f->sign;
    for (size_t i = 0; i < f->lu->rows; i++)
        d *= LINC_AT(f->lu, i, i);
    *det = d;
    return LINC_OK;
}

linc_status linc_lu_inverse(const linc_lu *f, linc_mat *inv)
{
    linc_status st = check_factor(f);
    if (st != LINC_OK)
        return st;
    if (!inv)
        return LINC_ERR_NULL;
    size_t n = f->lu->rows;
    if (inv->rows != n || inv->cols != n)
        return LINC_ERR_DIM;
    if (linc_mats_overlap(inv, f->lu))
        return LINC_ERR_ALIAS;
    if (f->singular)
        return LINC_ERR_SINGULAR;

    /* Solve A X = I. P * I is just the identity with permuted rows, so we
     * can write it directly instead of permuting a copy. */
    linc_mat_fill(inv, 0.0);
    for (size_t i = 0; i < n; i++)
        LINC_AT(inv, i, f->perm[i]) = 1.0;
    lu_substitute(f->lu, inv);
    return LINC_OK;
}

/* ------------------------------------------------------------------------ */
/* One-shot wrappers                                                         */
/* ------------------------------------------------------------------------ */

linc_status linc_solve(const linc_mat *A, const linc_vec *b, linc_vec *x)
{
    if (!A || !b || !x)
        return LINC_ERR_NULL;
    if (A->rows != A->cols || b->n != A->rows || x->n != A->rows)
        return LINC_ERR_DIM;
    linc_lu f;
    linc_status st = linc_lu_factor(A, &f);
    if (st != LINC_OK)
        return st;
    st = linc_lu_solve(&f, b, x);
    linc_lu_free(&f);
    return st;
}

linc_status linc_det(const linc_mat *A, double *det)
{
    if (!A || !det)
        return LINC_ERR_NULL;
    linc_lu f;
    linc_status st = linc_lu_factor(A, &f);
    if (st != LINC_OK)
        return st;
    st = linc_lu_det(&f, det);
    linc_lu_free(&f);
    return st;
}

linc_status linc_inverse(const linc_mat *A, linc_mat *inv)
{
    if (!A || !inv)
        return LINC_ERR_NULL;
    if (A->rows != A->cols || inv->rows != A->rows || inv->cols != A->cols)
        return LINC_ERR_DIM;
    linc_lu f;
    linc_status st = linc_lu_factor(A, &f);
    if (st != LINC_OK)
        return st;
    st = linc_lu_inverse(&f, inv);
    linc_lu_free(&f);
    return st;
}
