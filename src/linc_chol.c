/* Cholesky decomposition A = L * L^T for symmetric positive definite A.
 *
 * Writing out (L L^T)_ij = sum_k l_ik l_jk and solving for l_ij gives
 *
 *     l_jj = sqrt(a_jj - sum_{k<j} l_jk^2)
 *     l_ij = (a_ij - sum_{k<j} l_ik l_jk) / l_jj        for i > j
 *
 * Computing L one row at a time (the Cholesky-Banachiewicz order) makes both
 * sums dot products of two contiguous row prefixes. The square root argument
 * is positive for every j exactly when A is positive definite, so the
 * algorithm doubles as the cheapest available SPD test.
 *
 * Half the cost of LU (n^3/3 flops), no pivoting needed, and unconditionally
 * stable for SPD matrices. Use it for covariance matrices, normal equations,
 * Gram matrices, Hessians, and so on.
 */
#include "linc_internal.h"

#include <math.h>
#include <stdlib.h>

linc_status linc_cholesky(const linc_mat *A, linc_mat *L)
{
    if (!A || !L)
        return LINC_ERR_NULL;
    if (A->rows != A->cols || L->rows != A->rows || L->cols != A->cols)
        return LINC_ERR_DIM;
    if (L->data != A->data && linc_mats_overlap(L, A))
        return LINC_ERR_ALIAS;
    size_t n = A->rows;

    /* In place (L == A) works because l_ij is written to the slot of a_ij
     * right after a_ij is read, and only already-finished entries of L are
     * read afterwards. */
    for (size_t i = 0; i < n; i++) {
        const double *ai = &LINC_AT(A, i, 0);
        double *li = &LINC_AT(L, i, 0);
        for (size_t j = 0; j <= i; j++) {
            const double *lj = &LINC_AT(L, j, 0);
            double s = ai[j] - linc__dot(li, lj, j);
            if (i == j) {
                /* !(s > 0) also catches NaN. */
                if (!(s > 0.0) || !isfinite(s))
                    return LINC_ERR_NOT_SPD;
                li[i] = sqrt(s);
            } else {
                li[j] = s / lj[j];
            }
        }
        for (size_t j = i + 1; j < n; j++)
            li[j] = 0.0;
    }
    return LINC_OK;
}

linc_status linc_cholesky_solve(const linc_mat *L, const linc_vec *b,
                                linc_vec *x)
{
    if (!L || !b || !x)
        return LINC_ERR_NULL;
    if (L->rows != L->cols || b->n != L->rows || x->n != L->rows)
        return LINC_ERR_DIM;
    if (linc_overlaps(x->data, x->n, L->data, linc_mat_size(L)))
        return LINC_ERR_ALIAS;
    if (x->data != b->data && linc_overlaps(x->data, x->n, b->data, b->n))
        return LINC_ERR_ALIAS;

    /* A x = L (L^T x) = b: solve L y = b, then L^T x = y. */
    linc_status st = linc_vec_copy(b, x);
    if (st == LINC_OK)
        st = linc_trsv(LINC_LOWER, LINC_NO_TRANS, LINC_NON_UNIT, L, x);
    if (st == LINC_OK)
        st = linc_trsv(LINC_LOWER, LINC_TRANS, LINC_NON_UNIT, L, x);
    return st;
}
