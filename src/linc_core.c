/* Construction, level-1 vector kernels and elementwise matrix operations. */
#include "linc_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *linc_status_str(linc_status status)
{
    switch (status) {
    case LINC_OK: return "success";
    case LINC_ERR_NULL: return "null pointer argument";
    case LINC_ERR_DIM: return "dimension mismatch";
    case LINC_ERR_ALLOC: return "memory allocation failed";
    case LINC_ERR_ALIAS: return "output aliases an input";
    case LINC_ERR_SINGULAR: return "matrix is singular";
    case LINC_ERR_NOT_SPD: return "matrix is not symmetric positive definite";
    case LINC_ERR_NOT_SYMMETRIC: return "matrix is not symmetric";
    case LINC_ERR_RANK_DEFICIENT: return "matrix is rank deficient";
    case LINC_ERR_NO_CONVERGENCE: return "algorithm did not converge";
    case LINC_ERR_ARG: return "invalid argument";
    }
    return "unknown status";
}

/* ------------------------------------------------------------------------ */
/* Allocation                                                                */
/* ------------------------------------------------------------------------ */

/* Objects are allocated as one block: [header | padding | data]. The data
 * starts on a 64-byte boundary (a cache line, and wide enough for any SIMD
 * load), and a single free() releases everything. */
#define LINC_ALIGN 64

static void *alloc_block(size_t header, size_t count, double **data)
{
    size_t bytes;
    if (!linc_mul_ok(count, sizeof(double), &bytes))
        return NULL;
    if (bytes > SIZE_MAX - header - LINC_ALIGN)
        return NULL;
    unsigned char *base = calloc(1, header + LINC_ALIGN + bytes);
    if (!base)
        return NULL;
    uintptr_t p = (uintptr_t)(base + header);
    p = (p + LINC_ALIGN - 1) & ~(uintptr_t)(LINC_ALIGN - 1);
    *data = (double *)p;
    return base;
}

linc_vec *linc_vec_new(size_t n)
{
    double *data;
    linc_vec *v = alloc_block(sizeof(linc_vec), n, &data);
    if (!v)
        return NULL;
    v->n = n;
    v->data = data;
    return v;
}

linc_vec *linc_vec_from(const double *src, size_t n)
{
    if (!src && n > 0)
        return NULL;
    linc_vec *v = linc_vec_new(n);
    if (v && n > 0)
        memcpy(v->data, src, n * sizeof(double));
    return v;
}

linc_vec *linc_vec_clone(const linc_vec *v)
{
    return v ? linc_vec_from(v->data, v->n) : NULL;
}

void linc_vec_free(linc_vec *v)
{
    free(v);
}

linc_vec linc_vec_view(double *data, size_t n)
{
    linc_vec v = {n, data};
    return v;
}

linc_mat *linc_mat_new(size_t rows, size_t cols)
{
    size_t count;
    double *data;
    if (!linc_mul_ok(rows, cols, &count))
        return NULL;
    linc_mat *m = alloc_block(sizeof(linc_mat), count, &data);
    if (!m)
        return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = data;
    return m;
}

linc_mat *linc_mat_from(const double *src, size_t rows, size_t cols)
{
    linc_mat *m = linc_mat_new(rows, cols);
    if (!m)
        return NULL;
    if (linc_mat_size(m) > 0) {
        if (!src) {
            free(m);
            return NULL;
        }
        memcpy(m->data, src, linc_mat_size(m) * sizeof(double));
    }
    return m;
}

linc_mat *linc_mat_identity(size_t n)
{
    linc_mat *m = linc_mat_new(n, n);
    linc_mat_set_identity(m);
    return m;
}

linc_mat *linc_mat_clone(const linc_mat *m)
{
    return m ? linc_mat_from(m->data, m->rows, m->cols) : NULL;
}

void linc_mat_free(linc_mat *m)
{
    free(m);
}

linc_mat linc_mat_view(double *data, size_t rows, size_t cols)
{
    linc_mat m = {rows, cols, data};
    return m;
}

linc_status linc_vec_copy(const linc_vec *src, linc_vec *dst)
{
    if (!src || !dst)
        return LINC_ERR_NULL;
    if (src->n != dst->n)
        return LINC_ERR_DIM;
    if (src->data != dst->data && src->n > 0)
        memmove(dst->data, src->data, src->n * sizeof(double));
    return LINC_OK;
}

linc_status linc_mat_copy(const linc_mat *src, linc_mat *dst)
{
    if (!src || !dst)
        return LINC_ERR_NULL;
    if (src->rows != dst->rows || src->cols != dst->cols)
        return LINC_ERR_DIM;
    if (src->data != dst->data && linc_mat_size(src) > 0)
        memmove(dst->data, src->data, linc_mat_size(src) * sizeof(double));
    return LINC_OK;
}

void linc_vec_fill(linc_vec *v, double value)
{
    if (!v)
        return;
    for (size_t i = 0; i < v->n; i++)
        v->data[i] = value;
}

void linc_mat_fill(linc_mat *m, double value)
{
    if (!m)
        return;
    size_t n = linc_mat_size(m);
    for (size_t i = 0; i < n; i++)
        m->data[i] = value;
}

void linc_mat_set_identity(linc_mat *m)
{
    if (!m)
        return;
    linc_mat_fill(m, 0.0);
    size_t d = m->rows < m->cols ? m->rows : m->cols;
    for (size_t i = 0; i < d; i++)
        LINC_AT(m, i, i) = 1.0;
}

linc_status linc_mat_get_col(const linc_mat *m, size_t j, linc_vec *out)
{
    if (!m || !out)
        return LINC_ERR_NULL;
    if (j >= m->cols)
        return LINC_ERR_ARG;
    if (out->n != m->rows)
        return LINC_ERR_DIM;
    for (size_t i = 0; i < m->rows; i++)
        out->data[i] = LINC_AT(m, i, j);
    return LINC_OK;
}

linc_status linc_mat_set_col(linc_mat *m, size_t j, const linc_vec *v)
{
    if (!m || !v)
        return LINC_ERR_NULL;
    if (j >= m->cols)
        return LINC_ERR_ARG;
    if (v->n != m->rows)
        return LINC_ERR_DIM;
    for (size_t i = 0; i < m->rows; i++)
        LINC_AT(m, i, j) = v->data[i];
    return LINC_OK;
}

/* ------------------------------------------------------------------------ */
/* Level 1                                                                   */
/* ------------------------------------------------------------------------ */

linc_status linc_vec_dot(const linc_vec *x, const linc_vec *y, double *out)
{
    if (!x || !y || !out)
        return LINC_ERR_NULL;
    if (x->n != y->n)
        return LINC_ERR_DIM;
    *out = linc__dot(x->data, y->data, x->n);
    return LINC_OK;
}

linc_status linc_vec_axpy(double alpha, const linc_vec *x, linc_vec *y)
{
    if (!x || !y)
        return LINC_ERR_NULL;
    if (x->n != y->n)
        return LINC_ERR_DIM;
    if (alpha == 0.0)
        return LINC_OK;
    for (size_t i = 0; i < x->n; i++)
        y->data[i] += alpha * x->data[i];
    return LINC_OK;
}

void linc_vec_scale(linc_vec *x, double alpha)
{
    if (!x)
        return;
    for (size_t i = 0; i < x->n; i++)
        x->data[i] *= alpha;
}

linc_status linc_vec_add(const linc_vec *x, const linc_vec *y, linc_vec *out)
{
    if (!x || !y || !out)
        return LINC_ERR_NULL;
    if (x->n != y->n || x->n != out->n)
        return LINC_ERR_DIM;
    for (size_t i = 0; i < x->n; i++)
        out->data[i] = x->data[i] + y->data[i];
    return LINC_OK;
}

linc_status linc_vec_sub(const linc_vec *x, const linc_vec *y, linc_vec *out)
{
    if (!x || !y || !out)
        return LINC_ERR_NULL;
    if (x->n != y->n || x->n != out->n)
        return LINC_ERR_DIM;
    for (size_t i = 0; i < x->n; i++)
        out->data[i] = x->data[i] - y->data[i];
    return LINC_OK;
}

double linc__max_abs(const double *x, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > m || isnan(a))
            m = a;
        if (isnan(m))
            return m;
    }
    return m;
}

/* sqrt(sum x^2) computed naively overflows once any |x| > ~1e154 and loses
 * everything to underflow below ~1e-154. The fast path squares directly;
 * only if the result is out of the safe range do we redo the sum with a
 * running scale factor (the classic LAPACK dlassq recurrence):
 *     sum x^2 = scale^2 * ssq,  with scale = max |x| seen so far. */
double linc__norm2(const double *x, size_t n)
{
    double s = linc__dot(x, x, n);
    if (isfinite(s) && s >= DBL_MIN / DBL_EPSILON)
        return sqrt(s);
    if (isnan(s))
        return s;

    double scale = 0.0, ssq = 1.0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs(x[i]);
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
    return scale * sqrt(ssq);
}

linc_status linc_vec_norm2(const linc_vec *x, double *out)
{
    if (!x || !out)
        return LINC_ERR_NULL;
    *out = linc__norm2(x->data, x->n);
    return LINC_OK;
}

linc_status linc_vec_norm1(const linc_vec *x, double *out)
{
    if (!x || !out)
        return LINC_ERR_NULL;
    double s = 0.0;
    for (size_t i = 0; i < x->n; i++)
        s += fabs(x->data[i]);
    *out = s;
    return LINC_OK;
}

linc_status linc_vec_norm_inf(const linc_vec *x, double *out)
{
    if (!x || !out)
        return LINC_ERR_NULL;
    *out = linc__max_abs(x->data, x->n);
    return LINC_OK;
}

/* ------------------------------------------------------------------------ */
/* Whole-matrix operations                                                   */
/* ------------------------------------------------------------------------ */

linc_status linc_mat_transpose(const linc_mat *A, linc_mat *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    if (out->rows != A->cols || out->cols != A->rows)
        return LINC_ERR_DIM;

    if (out->data == A->data) {
        /* In place is only well defined for square matrices: swap across
         * the diagonal. */
        if (A->rows != A->cols)
            return LINC_ERR_ALIAS;
        for (size_t i = 0; i < A->rows; i++)
            for (size_t j = i + 1; j < A->cols; j++) {
                double t = LINC_AT(out, i, j);
                LINC_AT(out, i, j) = LINC_AT(out, j, i);
                LINC_AT(out, j, i) = t;
            }
        return LINC_OK;
    }
    if (linc_mats_overlap(A, out))
        return LINC_ERR_ALIAS;

    /* A naive transpose reads rows and writes columns, so one of the two
     * streams always strides through memory and misses cache. Working in
     * small tiles keeps both the source and destination tile resident.
     * Small really matters: with a power-of-two row stride every row of a
     * tile maps to the same cache set, so a 32 x 32 tile overflows the
     * set's associativity and thrashes. 8 x 8 measured 2-5x faster than 32
     * for n >= 1024 (see bench). */
    enum { TILE = 8 };
    for (size_t ii = 0; ii < A->rows; ii += TILE) {
        size_t ie = ii + TILE < A->rows ? ii + TILE : A->rows;
        for (size_t jj = 0; jj < A->cols; jj += TILE) {
            size_t je = jj + TILE < A->cols ? jj + TILE : A->cols;
            for (size_t i = ii; i < ie; i++)
                for (size_t j = jj; j < je; j++)
                    LINC_AT(out, j, i) = LINC_AT(A, i, j);
        }
    }
    return LINC_OK;
}

static linc_status mat_binop(const linc_mat *A, const linc_mat *B,
                             linc_mat *out, double sign)
{
    if (!A || !B || !out)
        return LINC_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols || A->rows != out->rows ||
        A->cols != out->cols)
        return LINC_ERR_DIM;
    size_t n = linc_mat_size(A);
    for (size_t i = 0; i < n; i++)
        out->data[i] = A->data[i] + sign * B->data[i];
    return LINC_OK;
}

linc_status linc_mat_add(const linc_mat *A, const linc_mat *B, linc_mat *out)
{
    return mat_binop(A, B, out, 1.0);
}

linc_status linc_mat_sub(const linc_mat *A, const linc_mat *B, linc_mat *out)
{
    return mat_binop(A, B, out, -1.0);
}

void linc_mat_scale(linc_mat *A, double alpha)
{
    if (!A)
        return;
    size_t n = linc_mat_size(A);
    for (size_t i = 0; i < n; i++)
        A->data[i] *= alpha;
}

linc_status linc_mat_trace(const linc_mat *A, double *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    if (A->rows != A->cols)
        return LINC_ERR_DIM;
    double t = 0.0;
    for (size_t i = 0; i < A->rows; i++)
        t += LINC_AT(A, i, i);
    *out = t;
    return LINC_OK;
}

linc_status linc_mat_norm(const linc_mat *A, linc_norm kind, double *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    double r = 0.0;
    switch (kind) {
    case LINC_NORM_FRO:
        r = linc__norm2(A->data, linc_mat_size(A));
        break;
    case LINC_NORM_MAX:
        r = linc__max_abs(A->data, linc_mat_size(A));
        break;
    case LINC_NORM_INF:
        for (size_t i = 0; i < A->rows; i++) {
            double s = 0.0;
            for (size_t j = 0; j < A->cols; j++)
                s += fabs(LINC_AT(A, i, j));
            if (s > r || isnan(s))
                r = s;
        }
        break;
    case LINC_NORM_1: {
        /* Column sums, accumulated row by row so the walk is contiguous. */
        double *sums = calloc(A->cols ? A->cols : 1, sizeof(double));
        if (!sums)
            return LINC_ERR_ALLOC;
        for (size_t i = 0; i < A->rows; i++)
            for (size_t j = 0; j < A->cols; j++)
                sums[j] += fabs(LINC_AT(A, i, j));
        r = linc__max_abs(sums, A->cols);
        free(sums);
        break;
    }
    default:
        return LINC_ERR_ARG;
    }
    *out = r;
    return LINC_OK;
}

linc_status linc_mat_max_abs_diff(const linc_mat *A, const linc_mat *B,
                                  double *out)
{
    if (!A || !B || !out)
        return LINC_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols)
        return LINC_ERR_DIM;
    double m = 0.0;
    size_t n = linc_mat_size(A);
    for (size_t i = 0; i < n; i++) {
        double d = fabs(A->data[i] - B->data[i]);
        if (d > m || isnan(d))
            m = d;
    }
    *out = m;
    return LINC_OK;
}

linc_status linc_mat_is_symmetric(const linc_mat *A, double tol, int *out)
{
    if (!A || !out)
        return LINC_ERR_NULL;
    if (tol < 0.0 || isnan(tol))
        return LINC_ERR_ARG;
    *out = 0;
    if (A->rows != A->cols)
        return LINC_OK;
    double bound = tol * linc__max_abs(A->data, linc_mat_size(A));
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = i + 1; j < A->cols; j++)
            if (!(fabs(LINC_AT(A, i, j) - LINC_AT(A, j, i)) <= bound))
                return LINC_OK;
    *out = 1;
    return LINC_OK;
}
