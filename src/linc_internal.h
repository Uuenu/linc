/* Private helpers shared by the implementation files. Not installed. */
#ifndef LINC_INTERNAL_H
#define LINC_INTERNAL_H

#include "linc.h"

#include <stdint.h>

/* True if [a, a+na) and [b, b+nb) overlap. Compared as integers because
 * relational comparison of pointers into different objects is undefined. */
static inline int linc_overlaps(const double *a, size_t na, const double *b,
                                size_t nb)
{
    if (na == 0 || nb == 0)
        return 0;
    uintptr_t a0 = (uintptr_t)a, a1 = (uintptr_t)(a + na);
    uintptr_t b0 = (uintptr_t)b, b1 = (uintptr_t)(b + nb);
    return a0 < b1 && b0 < a1;
}

static inline size_t linc_mat_size(const linc_mat *m)
{
    return m->rows * m->cols;
}

static inline int linc_mats_overlap(const linc_mat *a, const linc_mat *b)
{
    return linc_overlaps(a->data, linc_mat_size(a), b->data, linc_mat_size(b));
}

/* sum a[i] * b[i]. A single accumulator makes every add wait for the previous
 * one (a 3-4 cycle dependency chain), and the compiler may not reassociate
 * floating-point sums to fix that on its own. Eight independent partial sums
 * keep the FMA units busy and map directly onto 2- or 4-wide SIMD lanes. */
static inline double linc__dot(const double *restrict a,
                               const double *restrict b, size_t n)
{
    double s[8] = {0};
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (size_t l = 0; l < 8; l++)
            s[l] += a[i + l] * b[i + l];
    double t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; i < n; i++)
        t += a[i] * b[i];
    return t;
}

/* Overflow-safe sqrt(sum x[i]^2) over a contiguous array. */
double linc__norm2(const double *x, size_t n);

/* max |x[i]| over a contiguous array (0 for n == 0, NaN-propagating). */
double linc__max_abs(const double *x, size_t n);

/* Checked a * b; returns 0 and leaves *out untouched on overflow. */
static inline int linc_mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}

#endif /* LINC_INTERNAL_H */
