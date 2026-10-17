/* Principal component analysis of 2-D points.
 *
 * Centre the data, form the covariance matrix C = X^T X / (m - 1), and take
 * its eigendecomposition: eigenvectors are the principal axes, eigenvalues
 * the variance along each. Points here are scattered along the direction
 * (cos 30deg, sin 30deg), so that should come out as the first component.
 */
#include "linc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    enum { M = 500, D = 2 };
    const double angle = 30.0 * 3.14159265358979323846 / 180.0;
    linc_mat *X = linc_mat_new(M, D), *Xt = linc_mat_new(D, M);
    linc_mat *C = linc_mat_new(D, D), *V = linc_mat_new(D, D);
    linc_vec *w = linc_vec_new(D);
    if (!X || !Xt || !C || !V || !w)
        return EXIT_FAILURE;

    srand(1);
    for (size_t i = 0; i < M; i++) {
        double major = ((double)rand() / RAND_MAX - 0.5) * 10.0;
        double minor = ((double)rand() / RAND_MAX - 0.5) * 1.0;
        LINC_AT(X, i, 0) = 3.0 + major * cos(angle) - minor * sin(angle);
        LINC_AT(X, i, 1) = -1.0 + major * sin(angle) + minor * cos(angle);
    }

    /* Centre each column. */
    for (size_t j = 0; j < D; j++) {
        double mean = 0.0;
        for (size_t i = 0; i < M; i++)
            mean += LINC_AT(X, i, j);
        mean /= M;
        for (size_t i = 0; i < M; i++)
            LINC_AT(X, i, j) -= mean;
    }

    /* C = X^T X / (m - 1) */
    linc_mat_transpose(X, Xt);
    linc_gemm(1.0 / (M - 1), Xt, X, 0.0, C);

    linc_status st = linc_eig_sym(C, w, V);
    if (st != LINC_OK) {
        fprintf(stderr, "eig_sym: %s\n", linc_status_str(st));
        return EXIT_FAILURE;
    }

    /* Eigenvalues are ascending, so the principal axis is the last one. */
    double total = w->data[0] + w->data[1];
    for (size_t k = D; k-- > 0;) {
        double vx = LINC_AT(V, 0, k), vy = LINC_AT(V, 1, k);
        if (vx < 0) { /* eigenvectors are defined up to sign */
            vx = -vx;
            vy = -vy;
        }
        printf("PC%zu: direction (%.3f, %.3f) at %.1f deg, variance %.3f "
               "(%.1f%%)\n",
               D - k, vx, vy, atan2(vy, vx) * 180.0 / 3.14159265358979323846,
               w->data[k], 100.0 * w->data[k] / total);
    }

    linc_mat_free(X);
    linc_mat_free(Xt);
    linc_mat_free(C);
    linc_mat_free(V);
    linc_vec_free(w);
    return EXIT_SUCCESS;
}
