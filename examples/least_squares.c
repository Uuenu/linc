/* Fit a quadratic y = c0 + c1 t + c2 t^2 to noisy data by least squares.
 *
 * Each sample gives one row [1, t, t^2] of the m x 3 Vandermonde matrix A;
 * QR finds the c minimising ||A c - y||. (Solving the normal equations
 * A^T A c = A^T y instead squares the condition number and loses accuracy.)
 */
#include "linc.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    enum { M = 50 };
    const double true_c[3] = {1.0, -2.0, 0.5};
    linc_mat *A = linc_mat_new(M, 3);
    linc_vec *y = linc_vec_new(M), *c = linc_vec_new(3);
    if (!A || !y || !c)
        return EXIT_FAILURE;

    srand(7);
    for (size_t i = 0; i < M; i++) {
        double t = (double)i / (M - 1) * 4.0;
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        LINC_AT(A, i, 0) = 1.0;
        LINC_AT(A, i, 1) = t;
        LINC_AT(A, i, 2) = t * t;
        y->data[i] = true_c[0] + true_c[1] * t + true_c[2] * t * t + noise;
    }

    linc_status st = linc_lstsq(A, y, c);
    if (st != LINC_OK) {
        fprintf(stderr, "lstsq: %s\n", linc_status_str(st));
        return EXIT_FAILURE;
    }
    printf("fitted: y = %.4f %+.4f t %+.4f t^2\n", c->data[0], c->data[1],
           c->data[2]);
    printf("true:   y = %.4f %+.4f t %+.4f t^2\n", true_c[0], true_c[1],
           true_c[2]);

    /* RMS of the residual: should be close to the noise level (~0.058). */
    linc_vec *r = linc_vec_clone(y);
    double rn;
    linc_gemv(LINC_NO_TRANS, 1.0, A, c, -1.0, r);
    linc_vec_norm2(r, &rn);
    printf("rms residual = %.4f\n", rn / 7.0710678118654755); /* sqrt(50) */

    linc_vec_free(r);
    linc_mat_free(A);
    linc_vec_free(y);
    linc_vec_free(c);
    return EXIT_SUCCESS;
}
