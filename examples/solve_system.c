/* Solve a linear system, reuse the factorization, and check the residual.
 *
 *   2x +  y -  z =   8
 *  -3x -  y + 2z = -11
 *  -2x +  y + 2z =  -3        solution: x = 2, y = 3, z = -1
 */
#include "linc.h"

#include <stdio.h>
#include <stdlib.h>

#define TRY(expr)                                                            \
    do {                                                                     \
        linc_status st_ = (expr);                                            \
        if (st_ != LINC_OK) {                                                \
            fprintf(stderr, "%s: %s\n", #expr, linc_status_str(st_));        \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    double a[] = {2, 1, -1, -3, -1, 2, -2, 1, 2};
    double b[] = {8, -11, -3};
    double x[3], r[3];
    linc_mat A = linc_mat_view(a, 3, 3);
    linc_vec bv = linc_vec_view(b, 3), xv = linc_vec_view(x, 3),
             rv = linc_vec_view(r, 3);

    /* Factor once (O(n^3)), then every solve is only O(n^2). */
    linc_lu lu;
    TRY(linc_lu_factor(&A, &lu));
    TRY(linc_lu_solve(&lu, &bv, &xv));
    printf("x = [%g, %g, %g]\n", x[0], x[1], x[2]);

    double det;
    TRY(linc_lu_det(&lu, &det));
    printf("det(A) = %g\n", det);

    /* Residual r = A x - b should be at rounding-error level. */
    TRY(linc_vec_copy(&bv, &rv));
    TRY(linc_gemv(LINC_NO_TRANS, 1.0, &A, &xv, -1.0, &rv));
    double rn;
    TRY(linc_vec_norm2(&rv, &rn));
    printf("||Ax - b|| = %.3g\n", rn);

    linc_mat *inv = linc_mat_new(3, 3);
    if (!inv)
        return EXIT_FAILURE;
    TRY(linc_lu_inverse(&lu, inv));
    printf("inv(A) =\n");
    for (size_t i = 0; i < 3; i++)
        printf("  [%8.4f %8.4f %8.4f]\n", LINC_AT(inv, i, 0),
               LINC_AT(inv, i, 1), LINC_AT(inv, i, 2));

    linc_mat_free(inv);
    linc_lu_free(&lu);
    return EXIT_SUCCESS;
}
