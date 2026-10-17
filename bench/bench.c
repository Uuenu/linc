/* linc benchmarks.
 *
 *   bench_linc [--quick] [--csv] [--min-time SECONDS] [--only NAME]
 *
 * Every measurement repeats the operation until at least --min-time seconds
 * have elapsed (and at least 3 times) and reports the fastest run. Fastest
 * rather than mean, because noise (other processes, frequency ramps,
 * interrupts) only ever makes a run slower.
 *
 * GFLOP/s uses the conventional flop count for each algorithm (e.g. 2n^3 for
 * GEMM, 2n^3/3 for LU), so numbers are comparable with published BLAS and
 * LAPACK results.
 *
 * Build with BLAS=1 to add the system BLAS (Accelerate / OpenBLAS) dgemm as
 * a reference point. It is multithreaded and hand-tuned assembly, so it is a
 * ceiling, not a fair fight.
 */
#define _POSIX_C_SOURCE 200809L
#include "linc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef LINC_BENCH_CBLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

static double g_min_time = 0.25;
static int g_csv;
static const char *g_only;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t g_rng = 42;
static double rnd(void)
{
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(g_rng >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static linc_mat *rand_mat(size_t r, size_t c)
{
    linc_mat *m = linc_mat_new(r, c);
    if (!m) {
        fprintf(stderr, "out of memory\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < r * c; i++)
        m->data[i] = rnd();
    return m;
}

static linc_mat *rand_spd(size_t n)
{
    linc_mat *M = rand_mat(n, n), *S = linc_mat_new(n, n);
    linc_mat *Mt = linc_mat_new(n, n);
    linc_mat_transpose(M, Mt);
    linc_mat_mul(Mt, M, S);
    for (size_t i = 0; i < n; i++)
        LINC_AT(S, i, i) += (double)n;
    linc_mat_free(M);
    linc_mat_free(Mt);
    return S;
}

static void must(linc_status st, const char *what)
{
    if (st != LINC_OK) {
        fprintf(stderr, "%s failed: %s\n", what, linc_status_str(st));
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------------------------------ */
/* Timing harness                                                            */
/* ------------------------------------------------------------------------ */

typedef void (*bench_fn)(void *ctx);

/* Best wall time of one call to fn(ctx), in seconds.
 *
 * Fast operations are timed in batches: some clocks (macOS) only tick once
 * per microsecond, so a 3 us call timed on its own would be mostly
 * quantisation error. The batch grows until one batch takes >= 2 ms. */
static double measure(bench_fn fn, void *ctx)
{
    fn(ctx); /* warm-up: page faults, caches, CPU frequency */
    long batch = 1;
    for (;;) {
        double t0 = now();
        for (long i = 0; i < batch; i++)
            fn(ctx);
        if (now() - t0 >= 2e-3 || batch >= (1L << 24))
            break;
        batch *= 2;
    }

    double best = INFINITY, start = now();
    int reps = 0;
    do {
        double t0 = now();
        for (long i = 0; i < batch; i++)
            fn(ctx);
        double dt = (now() - t0) / (double)batch;
        if (dt < best)
            best = dt;
        reps++;
    } while (reps < 3 || now() - start < g_min_time);
    return best;
}

static void header(const char *title)
{
    if (g_csv)
        return;
    printf("\n%s\n", title);
    printf("  %-22s %6s %12s %10s %10s\n", "operation", "n", "time", "GFLOP/s",
           "note");
}

static void report(const char *op, size_t n, double secs, double flops,
                   const char *note)
{
    double gflops = flops > 0 ? flops / secs * 1e-9 : 0.0;
    if (g_csv) {
        printf("%s,%zu,%.9f,%.3f,%s\n", op, n, secs, gflops, note ? note : "");
    } else {
        char t[32];
        if (secs < 1e-3)
            snprintf(t, sizeof t, "%.2f us", secs * 1e6);
        else if (secs < 1.0)
            snprintf(t, sizeof t, "%.2f ms", secs * 1e3);
        else
            snprintf(t, sizeof t, "%.2f s", secs);
        if (flops > 0)
            printf("  %-22s %6zu %12s %10.2f %10s\n", op, n, t, gflops,
                   note ? note : "");
        else
            printf("  %-22s %6zu %12s %10s %10s\n", op, n, t, "-",
                   note ? note : "");
    }
    fflush(stdout);
}

static int enabled(const char *group)
{
    return !g_only || strcmp(g_only, group) == 0;
}

/* ------------------------------------------------------------------------ */
/* GEMM                                                                      */
/* ------------------------------------------------------------------------ */

typedef struct {
    linc_gemm_algo algo;
    linc_mat *A, *B, *C;
} gemm_ctx;

static void run_gemm(void *p)
{
    gemm_ctx *c = p;
    must(linc_gemm_ex(c->algo, 1.0, c->A, c->B, 0.0, c->C), "gemm");
}

#ifdef LINC_BENCH_CBLAS
static void run_cblas(void *p)
{
    gemm_ctx *c = p;
    int n = (int)c->A->rows;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                c->A->data, n, c->B->data, n, 0.0, c->C->data, n);
}
#endif

static void bench_gemm(int quick)
{
    static const size_t full[] = {64, 128, 256, 512, 1024, 2048};
    static const size_t small[] = {64, 128, 256};
    const size_t *sizes = quick ? small : full;
    size_t count = quick ? 3 : 6;
    static const struct {
        linc_gemm_algo algo;
        const char *name;
        size_t max_n; /* the slow ones would take minutes at 2048 */
    } algos[] = {
        {LINC_GEMM_NAIVE, "gemm naive (ijk)", 1024},
        {LINC_GEMM_REORDERED, "gemm reordered (ikj)", 2048},
        {LINC_GEMM_BLOCKED, "gemm blocked", 2048},
        {LINC_GEMM_PACKED, "gemm packed", 2048},
    };

    header("GEMM  C = A * B  (square, 2n^3 flops; note = speedup vs naive)");
    for (size_t s = 0; s < count; s++) {
        size_t n = sizes[s];
        gemm_ctx c = {LINC_GEMM_AUTO, rand_mat(n, n), rand_mat(n, n),
                      linc_mat_new(n, n)};
        double flops = 2.0 * (double)n * (double)n * (double)n;
        double t_naive = 0.0;
        for (size_t a = 0; a < sizeof algos / sizeof algos[0]; a++) {
            if (n > algos[a].max_n)
                continue;
            c.algo = algos[a].algo;
            double t = measure(run_gemm, &c);
            char note[32] = "";
            if (algos[a].algo == LINC_GEMM_NAIVE)
                t_naive = t;
            else if (t_naive > 0)
                snprintf(note, sizeof note, "%.1fx", t_naive / t);
            report(algos[a].name, n, t, flops, note);
        }
#ifdef LINC_BENCH_CBLAS
        double t = measure(run_cblas, &c);
        char note[32] = "";
        if (t_naive > 0)
            snprintf(note, sizeof note, "%.1fx", t_naive / t);
        report("gemm system BLAS", n, t, flops, note);
#endif
        linc_mat_free(c.A);
        linc_mat_free(c.B);
        linc_mat_free(c.C);
    }
}

/* ------------------------------------------------------------------------ */
/* Level 1 / 2 and memory-bound operations                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    linc_mat *A, *B;
    linc_vec *x, *y;
    double sink;
} misc_ctx;

static void run_dot(void *p)
{
    misc_ctx *c = p;
    must(linc_vec_dot(c->x, c->y, &c->sink), "dot");
}
static void run_axpy(void *p)
{
    misc_ctx *c = p;
    must(linc_vec_axpy(1e-9, c->x, c->y), "axpy");
}
static void run_norm2(void *p)
{
    misc_ctx *c = p;
    must(linc_vec_norm2(c->x, &c->sink), "norm2");
}
static void run_gemv(void *p)
{
    misc_ctx *c = p;
    must(linc_gemv(LINC_NO_TRANS, 1.0, c->A, c->x, 0.0, c->y), "gemv");
}
static void run_gemv_t(void *p)
{
    misc_ctx *c = p;
    must(linc_gemv(LINC_TRANS, 1.0, c->A, c->x, 0.0, c->y), "gemv^T");
}
static void run_transpose(void *p)
{
    misc_ctx *c = p;
    must(linc_mat_transpose(c->A, c->B), "transpose");
}

static void bench_level12(int quick)
{
    header("Level 1 / 2  (memory-bound: GFLOP/s is limited by bandwidth)");
    size_t vn = quick ? 100000 : 1000000;
    misc_ctx c = {0};
    c.x = linc_vec_new(vn);
    c.y = linc_vec_new(vn);
    for (size_t i = 0; i < vn; i++) {
        c.x->data[i] = rnd();
        c.y->data[i] = rnd();
    }
    report("dot", vn, measure(run_dot, &c), 2.0 * (double)vn, NULL);
    report("axpy", vn, measure(run_axpy, &c), 2.0 * (double)vn, NULL);
    report("norm2", vn, measure(run_norm2, &c), 2.0 * (double)vn, NULL);
    linc_vec_free(c.x);
    linc_vec_free(c.y);

    static const size_t sizes[] = {256, 1024, 4096};
    for (size_t s = 0; s < (quick ? 2u : 3u); s++) {
        size_t n = sizes[s];
        c.A = rand_mat(n, n);
        c.B = linc_mat_new(n, n);
        c.x = linc_vec_new(n);
        c.y = linc_vec_new(n);
        for (size_t i = 0; i < n; i++)
            c.x->data[i] = rnd();
        double fl = 2.0 * (double)n * (double)n;
        report("gemv", n, measure(run_gemv, &c), fl, NULL);
        report("gemv transposed", n, measure(run_gemv_t, &c), fl, NULL);
        /* Bandwidth: 16 bytes moved per element (one read, one write). */
        double t = measure(run_transpose, &c);
        char note[32];
        snprintf(note, sizeof note, "%.1f GB/s",
                 16.0 * (double)n * (double)n / t * 1e-9);
        report("transpose", n, t, 0, note);
        linc_mat_free(c.A);
        linc_mat_free(c.B);
        linc_vec_free(c.x);
        linc_vec_free(c.y);
    }
}

/* ------------------------------------------------------------------------ */
/* Decompositions                                                            */
/* ------------------------------------------------------------------------ */

typedef struct {
    linc_mat *A, *out;
    linc_vec *b, *x;
} dec_ctx;

static void run_lu(void *p)
{
    dec_ctx *c = p;
    linc_lu f;
    must(linc_lu_factor(c->A, &f), "lu");
    linc_lu_free(&f);
}
static void run_solve(void *p)
{
    dec_ctx *c = p;
    must(linc_solve(c->A, c->b, c->x), "solve");
}
static void run_inverse(void *p)
{
    dec_ctx *c = p;
    must(linc_inverse(c->A, c->out), "inverse");
}
static void run_det(void *p)
{
    dec_ctx *c = p;
    double d;
    must(linc_det(c->A, &d), "det");
}
static void run_chol(void *p)
{
    dec_ctx *c = p;
    must(linc_cholesky(c->A, c->out), "cholesky");
}
static void run_qr(void *p)
{
    dec_ctx *c = p;
    linc_qr f;
    must(linc_qr_factor(c->A, &f), "qr");
    linc_qr_free(&f);
}
static void run_lstsq(void *p)
{
    dec_ctx *c = p;
    must(linc_lstsq(c->A, c->b, c->x), "lstsq");
}
static void run_eig(void *p)
{
    dec_ctx *c = p;
    must(linc_eig_sym(c->A, c->x, c->out), "eig_sym");
}
static void run_eigvals(void *p)
{
    dec_ctx *c = p;
    must(linc_eig_sym(c->A, c->x, NULL), "eig_sym");
}

static void bench_decomp(int quick)
{
    header("Decompositions and solvers  (square n x n)");
    static const size_t sizes[] = {100, 250, 500, 1000};
    size_t count = quick ? 2 : 4;
    for (size_t s = 0; s < count; s++) {
        size_t n = sizes[s];
        double n3 = (double)n * (double)n * (double)n;
        dec_ctx c = {rand_mat(n, n), linc_mat_new(n, n), linc_vec_new(n),
                     linc_vec_new(n)};
        for (size_t i = 0; i < n; i++)
            c.b->data[i] = rnd();

        report("lu factor", n, measure(run_lu, &c), 2.0 / 3.0 * n3, NULL);
        report("solve (lu)", n, measure(run_solve, &c),
               2.0 / 3.0 * n3 + 2.0 * (double)n * (double)n, NULL);
        report("det (lu)", n, measure(run_det, &c), 2.0 / 3.0 * n3, NULL);
        report("inverse (lu)", n, measure(run_inverse, &c), 2.0 * n3, NULL);
        report("qr factor", n, measure(run_qr, &c), 4.0 / 3.0 * n3, NULL);
        report("lstsq (qr)", n, measure(run_lstsq, &c), 4.0 / 3.0 * n3, NULL);

        linc_mat_free(c.A);
        c.A = rand_spd(n);
        report("cholesky", n, measure(run_chol, &c), 1.0 / 3.0 * n3, NULL);

        linc_mat_free(c.A);
        linc_mat_free(c.out);
        linc_vec_free(c.b);
        linc_vec_free(c.x);
    }

    /* Jacobi is O(n^3) per sweep with ~6-10 sweeps; keep sizes modest.
     * No standard flop count exists, so only time is reported. */
    header("Symmetric eigendecomposition (cyclic Jacobi)");
    static const size_t esizes[] = {50, 100, 200, 400};
    for (size_t s = 0; s < (quick ? 2u : 4u); s++) {
        size_t n = esizes[s];
        dec_ctx c = {rand_spd(n), linc_mat_new(n, n), NULL, linc_vec_new(n)};
        report("eig_sym values", n, measure(run_eigvals, &c), 0, NULL);
        report("eig_sym values+vectors", n, measure(run_eig, &c), 0, NULL);
        linc_mat_free(c.A);
        linc_mat_free(c.out);
        linc_vec_free(c.x);
    }
}

int main(int argc, char **argv)
{
    int quick = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0)
            quick = 1;
        else if (strcmp(argv[i], "--csv") == 0)
            g_csv = 1;
        else if (strcmp(argv[i], "--min-time") == 0 && i + 1 < argc)
            g_min_time = atof(argv[++i]);
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc)
            g_only = argv[++i];
        else {
            int help = strcmp(argv[i], "--help") == 0;
            fprintf(help ? stdout : stderr,
                    "usage: %s [--quick] [--csv] [--min-time S] "
                    "[--only gemm|level12|decomp]\n",
                    argv[0]);
            return help ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    if (quick && g_min_time > 0.05)
        g_min_time = 0.05;

    if (g_csv)
        printf("operation,n,seconds,gflops,note\n");
    else
        printf("linc %s benchmarks (single-threaded, best of N, min %.2fs "
               "per measurement)\n",
               LINC_VERSION, g_min_time);

    if (enabled("gemm"))
        bench_gemm(quick);
    if (enabled("level12"))
        bench_level12(quick);
    if (enabled("decomp"))
        bench_decomp(quick);
    return EXIT_SUCCESS;
}
