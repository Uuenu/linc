/* Householder QR decomposition and least squares.
 *
 * A Householder reflector H = I - tau * v * v^T is an orthogonal matrix that
 * mirrors vectors across the hyperplane orthogonal to v. For any vector x we
 * can choose v so that H x = beta * e_1, i.e. all entries but the first are
 * zeroed. Applying one reflector per column,
 *
 *     H_{n-1} ... H_1 H_0 A = R      =>      A = Q R,  Q = H_0 H_1 ... H_{n-1}
 *
 * Unlike Gram-Schmidt, the computed Q stays orthogonal to machine precision
 * regardless of conditioning, which is why QR is the standard way to solve
 * least-squares problems: ||A x - b|| = ||R x - Q^T b||, minimised by
 * back substitution on the top n rows.
 *
 * Storage follows LAPACK (dgeqrf): R overwrites the upper triangle, and each
 * v (scaled so v_0 = 1, which therefore need not be stored) goes below the
 * diagonal in its column. Q is never formed unless asked for.
 */
#include "linc_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Apply H_k = I - tau v v^T (v stored in column k of qr, rows k..m-1, with
 * implicit v_k = 1) to rows k..m-1, columns c0..c1-1 of M.
 *     M <- M - tau * v * (v^T M)
 * w = v^T M is accumulated row by row so all inner loops are contiguous. */
static void apply_reflector(const linc_mat *qr, size_t k, double tau,
                            linc_mat *M, size_t c0, size_t c1, double *w)
{
    if (tau == 0.0 || c0 >= c1)
        return;
    size_t m = qr->rows, len = c1 - c0;
    memcpy(w, &LINC_AT(M, k, c0), len * sizeof(double));
    for (size_t i = k + 1; i < m; i++) {
        double vi = LINC_AT(qr, i, k);
        const double *mi = &LINC_AT(M, i, c0);
        for (size_t j = 0; j < len; j++)
            w[j] += vi * mi[j];
    }
    double *mk = &LINC_AT(M, k, c0);
    for (size_t j = 0; j < len; j++)
        mk[j] -= tau * w[j];
    for (size_t i = k + 1; i < m; i++) {
        double s = tau * LINC_AT(qr, i, k);
        double *mi = &LINC_AT(M, i, c0);
        for (size_t j = 0; j < len; j++)
            mi[j] -= s * w[j];
    }
}

linc_status linc_qr_factor(const linc_mat *A, linc_qr *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    size_t m = A->rows, n = A->cols;
    if (m < n)
        return LINC_ERR_DIM;

    linc_mat *qr = linc_mat_clone(A);
    double *tau = calloc(n ? n : 1, sizeof(double));
    double *w = malloc((n ? n : 1) * sizeof(double));
    double *col = malloc((m ? m : 1) * sizeof(double));
    if (!qr || !tau || !w || !col) {
        linc_mat_free(qr);
        free(tau);
        free(w);
        free(col);
        return LINC_ERR_ALLOC;
    }

    for (size_t k = 0; k < n; k++) {
        /* x = A[k:m, k]; gather the tail to compute its norm safely. */
        size_t len = m - k - 1;
        for (size_t i = 0; i < len; i++)
            col[i] = LINC_AT(qr, k + 1 + i, k);
        double alpha = LINC_AT(qr, k, k);
        double xnorm = linc__norm2(col, len);

        if (xnorm == 0.0) {
            tau[k] = 0.0; /* already zero below the diagonal: H = I */
            continue;
        }
        /* beta = -sign(alpha) * ||x||. Picking the sign opposite to alpha
         * makes alpha - beta an addition of like-signed numbers, avoiding
         * catastrophic cancellation. */
        double beta = -copysign(hypot(alpha, xnorm), alpha);
        tau[k] = (beta - alpha) / beta;
        double scale = 1.0 / (alpha - beta);
        for (size_t i = k + 1; i < m; i++)
            LINC_AT(qr, i, k) *= scale;
        LINC_AT(qr, k, k) = beta;

        /* Apply H_k to the remaining columns. */
        apply_reflector(qr, k, tau[k], qr, k + 1, n, w);
    }

    free(w);
    free(col);
    out->qr = qr;
    out->tau = tau;
    return LINC_OK;
}

void linc_qr_free(linc_qr *f)
{
    if (!f)
        return;
    linc_mat_free(f->qr);
    free(f->tau);
    f->qr = NULL;
    f->tau = NULL;
}

linc_status linc_qr_r(const linc_qr *f, linc_mat *R)
{
    if (!f || !f->qr || !R)
        return LINC_ERR_NULL;
    size_t n = f->qr->cols;
    if (R->rows != n || R->cols != n)
        return LINC_ERR_DIM;
    if (linc_mats_overlap(R, f->qr))
        return LINC_ERR_ALIAS;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            LINC_AT(R, i, j) = j >= i ? LINC_AT(f->qr, i, j) : 0.0;
    return LINC_OK;
}

linc_status linc_qr_q(const linc_qr *f, linc_mat *Q)
{
    if (!f || !f->qr || !f->tau || !Q)
        return LINC_ERR_NULL;
    size_t m = f->qr->rows, n = f->qr->cols;
    if (Q->rows != m || Q->cols != n)
        return LINC_ERR_DIM;
    if (linc_mats_overlap(Q, f->qr))
        return LINC_ERR_ALIAS;
    double *w = malloc((n ? n : 1) * sizeof(double));
    if (!w)
        return LINC_ERR_ALLOC;

    /* Q = H_0 ... H_{n-1} * I[:, 0:n]. Applying the reflectors in reverse
     * means H_k only ever meets columns k..n-1 that are nonzero in rows
     * k..m-1, so each step touches a shrinking corner. */
    linc_mat_set_identity(Q);
    for (size_t k = n; k-- > 0;)
        apply_reflector(f->qr, k, f->tau[k], Q, k, n, w);
    free(w);
    return LINC_OK;
}

linc_status linc_qr_solve(const linc_qr *f, const linc_vec *b, linc_vec *x)
{
    if (!f || !f->qr || !f->tau || !b || !x)
        return LINC_ERR_NULL;
    const linc_mat *qr = f->qr;
    size_t m = qr->rows, n = qr->cols;
    if (b->n != m || x->n != n)
        return LINC_ERR_DIM;
    if (linc_overlaps(x->data, x->n, qr->data, linc_mat_size(qr)))
        return LINC_ERR_ALIAS;

    /* Rank check: a diagonal entry of R that is tiny relative to the
     * largest one means a column of A is (nearly) a combination of the
     * others and the minimiser is not unique. */
    double rmax = 0.0;
    for (size_t i = 0; i < n; i++)
        rmax = fmax(rmax, fabs(LINC_AT(qr, i, i)));
    double tol = (double)m * DBL_EPSILON * rmax;
    for (size_t i = 0; i < n; i++)
        if (!(fabs(LINC_AT(qr, i, i)) > tol))
            return LINC_ERR_RANK_DEFICIENT;

    /* y = Q^T b = H_{n-1} ... H_0 b, on a copy of b (an m x 1 matrix). */
    linc_mat *y = linc_mat_from(b->data, m, 1);
    if (!y)
        return LINC_ERR_ALLOC;
    double w;
    for (size_t k = 0; k < n; k++)
        apply_reflector(qr, k, f->tau[k], y, 0, 1, &w);

    /* R x = y[0:n]. In row-major storage with cols == n, the first n rows
     * of qr are a contiguous n x n matrix whose upper triangle is R. */
    memcpy(x->data, y->data, n * sizeof(double));
    linc_mat_free(y);
    linc_mat R = linc_mat_view(qr->data, n, n);
    return linc_trsv(LINC_UPPER, LINC_NO_TRANS, LINC_NON_UNIT, &R, x);
}

linc_status linc_lstsq(const linc_mat *A, const linc_vec *b, linc_vec *x)
{
    if (!A || !b || !x)
        return LINC_ERR_NULL;
    if (A->rows < A->cols || b->n != A->rows || x->n != A->cols)
        return LINC_ERR_DIM;
    linc_qr f;
    linc_status st = linc_qr_factor(A, &f);
    if (st != LINC_OK)
        return st;
    st = linc_qr_solve(&f, b, x);
    linc_qr_free(&f);
    return st;
}
