/* Symmetric eigendecomposition by the cyclic Jacobi method.
 *
 * A Jacobi rotation J(p, q, theta) is the identity except for a 2 x 2 block
 * [c s; -s c] in rows/columns p and q. Choosing theta so that the (p, q)
 * entry of A' = J^T A J becomes zero, and sweeping over all p < q repeatedly,
 * drives the off-diagonal mass to zero: each rotation reduces
 * off(A)^2 = sum_{i != j} a_ij^2 by exactly 2 a_pq^2. The diagonal then holds
 * the eigenvalues and the accumulated product of rotations V = J_1 J_2 ...
 * holds the eigenvectors.
 *
 * Jacobi is slower than the tridiagonal QR algorithm LAPACK uses (roughly
 * 5-10x for large n) but it is short, obviously correct, converges
 * quadratically once off(A) is small, and computes small eigenvalues to high
 * relative accuracy. Practical for n up to a few hundred.
 */
#include "linc_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define LINC_JACOBI_MAX_SWEEPS 100

static double off_diag_norm(const linc_mat *A)
{
    size_t n = A->rows;
    double scale = 0.0, ssq = 1.0;
    /* Same scaled sum of squares as linc__norm2, over the strict upper
     * triangle (A stays symmetric, so that is half of off(A)^2). */
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++) {
            double a = fabs(LINC_AT(A, i, j));
            if (a == 0.0)
                continue;
            if (scale < a) {
                double r = scale / a;
                ssq = 1.0 + ssq * r * r;
                scale = a;
            } else {
                double r = a / scale;
                ssq += r * r;
            }
        }
    return scale * sqrt(2.0 * ssq);
}

/* Zero a_pq (and a_qp) with one rotation, accumulating it into V. */
static void rotate(linc_mat *A, linc_mat *V, size_t p, size_t q)
{
    size_t n = A->rows;
    double apq = LINC_AT(A, p, q);
    double app = LINC_AT(A, p, p), aqq = LINC_AT(A, q, q);

    /* We need (c^2 - s^2) a_pq + c s (a_pp - a_qq) = 0. With t = s / c and
     * theta = (a_qq - a_pp) / (2 a_pq) that is t^2 + 2 theta t - 1 = 0.
     * Taking the smaller root (|t| <= 1, rotation angle <= pi/4) is what
     * guarantees convergence; this form of it avoids cancellation, and
     * hypot avoids overflow when theta is huge. */
    double theta = (aqq - app) / (2.0 * apq);
    double t = copysign(1.0, theta) / (fabs(theta) + hypot(theta, 1.0));
    double c = 1.0 / hypot(t, 1.0);
    double s = t * c;

    /* A <- A J (columns p and q), then A <- J^T A (rows p and q). */
    for (size_t k = 0; k < n; k++) {
        double akp = LINC_AT(A, k, p), akq = LINC_AT(A, k, q);
        LINC_AT(A, k, p) = c * akp - s * akq;
        LINC_AT(A, k, q) = s * akp + c * akq;
    }
    double *rp = &LINC_AT(A, p, 0), *rq = &LINC_AT(A, q, 0);
    for (size_t k = 0; k < n; k++) {
        double apk = rp[k], aqk = rq[k];
        rp[k] = c * apk - s * aqk;
        rq[k] = s * apk + c * aqk;
    }
    /* Exact zeros where the rotation was aimed, rather than rounding noise. */
    LINC_AT(A, p, q) = 0.0;
    LINC_AT(A, q, p) = 0.0;

    if (V)
        for (size_t k = 0; k < n; k++) {
            double vkp = LINC_AT(V, k, p), vkq = LINC_AT(V, k, q);
            LINC_AT(V, k, p) = c * vkp - s * vkq;
            LINC_AT(V, k, q) = s * vkp + c * vkq;
        }
}

linc_status linc_eig_sym(const linc_mat *A, linc_vec *w, linc_mat *V)
{
    if (!A || !w)
        return LINC_ERR_NULL;
    size_t n = A->rows;
    if (A->cols != n || w->n != n)
        return LINC_ERR_DIM;
    if (V && (V->rows != n || V->cols != n))
        return LINC_ERR_DIM;
    if (V && linc_overlaps(V->data, linc_mat_size(V), w->data, w->n))
        return LINC_ERR_ALIAS;

    int sym;
    linc_status st = linc_mat_is_symmetric(A, 1e3 * DBL_EPSILON, &sym);
    if (st != LINC_OK)
        return st;
    if (!sym)
        return LINC_ERR_NOT_SYMMETRIC;

    /* Work on a copy; V may alias A, so take the copy before touching V. */
    linc_mat *W = linc_mat_clone(A);
    if (!W)
        return LINC_ERR_ALLOC;
    if (V)
        linc_mat_set_identity(V);

    double fro = linc__norm2(W->data, n * n);
    double target = DBL_EPSILON * fro;
    int converged = 0;

    for (int sweep = 0; sweep < LINC_JACOBI_MAX_SWEEPS; sweep++) {
        double off = off_diag_norm(W);
        if (off <= target) {
            converged = 1;
            break;
        }
        /* Skipping entries that are already negligible next to their
         * diagonal saves work in late sweeps and avoids pointless rotations
         * that only add rounding error. */
        for (size_t p = 0; p + 1 < n; p++)
            for (size_t q = p + 1; q < n; q++) {
                double apq = fabs(LINC_AT(W, p, q));
                if (apq == 0.0)
                    continue;
                double dp = fabs(LINC_AT(W, p, p)), dq = fabs(LINC_AT(W, q, q));
                if (sweep > 3 && dp + 100.0 * apq == dp &&
                    dq + 100.0 * apq == dq) {
                    LINC_AT(W, p, q) = LINC_AT(W, q, p) = 0.0;
                    continue;
                }
                rotate(W, V, p, q);
            }
    }
    if (!converged && off_diag_norm(W) <= target)
        converged = 1;

    if (converged) {
        for (size_t i = 0; i < n; i++)
            w->data[i] = LINC_AT(W, i, i);

        /* Selection sort into ascending order, permuting V's columns along.
         * O(n^2) swaps of length n is noise next to the O(n^3) sweeps. */
        for (size_t i = 0; i + 1 < n; i++) {
            size_t min = i;
            for (size_t j = i + 1; j < n; j++)
                if (w->data[j] < w->data[min])
                    min = j;
            if (min == i)
                continue;
            double t = w->data[i];
            w->data[i] = w->data[min];
            w->data[min] = t;
            if (V)
                for (size_t k = 0; k < n; k++) {
                    double v = LINC_AT(V, k, i);
                    LINC_AT(V, k, i) = LINC_AT(V, k, min);
                    LINC_AT(V, k, min) = v;
                }
        }
    }
    linc_mat_free(W);
    return converged ? LINC_OK : LINC_ERR_NO_CONVERGENCE;
}
