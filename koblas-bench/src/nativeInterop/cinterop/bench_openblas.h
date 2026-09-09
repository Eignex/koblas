#pragma once

#include <dlfcn.h>
#include <stddef.h>

static inline void *koblas_openblas_handle(void) {
    static void *handle;
    static int attempted;
    if (!attempted) {
        attempted = 1;
        handle = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
        if (handle == NULL) handle = dlopen("libopenblas.so", RTLD_LAZY | RTLD_LOCAL);
    }
    return handle;
}

static inline int koblas_openblas_available(void) {
    static const char *required[] = {
        "cblas_ddot", "cblas_daxpy", "cblas_dscal", "cblas_dnrm2", "cblas_dasum", "cblas_dswap",
        "cblas_drotm", "cblas_drot", "cblas_dgemv", "cblas_dsymv", "cblas_dger", "cblas_dsyr",
        "cblas_dsyr2", "cblas_dtrsv", "cblas_dtrmv", "cblas_dgemm", "cblas_dsyrk", "cblas_dsyr2k",
        "cblas_dsymm", "cblas_dtrsm", "cblas_dtrmm", "cblas_dgemmt", "openblas_set_num_threads"
    };
    void *handle = koblas_openblas_handle();
    if (handle == NULL) return 0;
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (dlsym(handle, required[i]) == NULL) return 0;
    }
    return 1;
}

static inline const char *koblas_openblas_version(void) {
    typedef const char *(*function_t)(void);
    function_t function = (function_t)dlsym(koblas_openblas_handle(), "openblas_get_config");
    return function == NULL ? "unknown" : function();
}

#define KOBLAS_RESOLVE(name, type) ({ \
    static type function; \
    if (function == NULL) function = (type)dlsym(koblas_openblas_handle(), #name); \
    function; \
})

static inline double cblas_ddot(int n, const double *x, int ix, const double *y, int iy) {
    typedef double (*function_t)(int, const double *, int, const double *, int);
    return KOBLAS_RESOLVE(cblas_ddot, function_t)(n, x, ix, y, iy);
}
static inline void cblas_daxpy(int n, double a, const double *x, int ix, double *y, int iy) {
    typedef void (*function_t)(int, double, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_daxpy, function_t)(n, a, x, ix, y, iy);
}
static inline void cblas_dscal(int n, double a, double *x, int ix) {
    typedef void (*function_t)(int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dscal, function_t)(n, a, x, ix);
}
static inline double cblas_dnrm2(int n, const double *x, int ix) {
    typedef double (*function_t)(int, const double *, int);
    return KOBLAS_RESOLVE(cblas_dnrm2, function_t)(n, x, ix);
}
static inline double cblas_dasum(int n, const double *x, int ix) {
    typedef double (*function_t)(int, const double *, int);
    return KOBLAS_RESOLVE(cblas_dasum, function_t)(n, x, ix);
}
static inline void cblas_dswap(int n, double *x, int ix, double *y, int iy) {
    typedef void (*function_t)(int, double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dswap, function_t)(n, x, ix, y, iy);
}
static inline void cblas_drotm(int n, double *x, int ix, double *y, int iy, const double *p) {
    typedef void (*function_t)(int, double *, int, double *, int, const double *);
    KOBLAS_RESOLVE(cblas_drotm, function_t)(n, x, ix, y, iy, p);
}
static inline void cblas_drot(int n, double *x, int ix, double *y, int iy, double c, double s) {
    typedef void (*function_t)(int, double *, int, double *, int, double, double);
    KOBLAS_RESOLVE(cblas_drot, function_t)(n, x, ix, y, iy, c, s);
}
static inline void cblas_dgemv(int o, int t, int m, int n, double a, const double *x, int ld, const double *y, int iy, double b, double *z, int iz) {
    typedef void (*function_t)(int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dgemv, function_t)(o, t, m, n, a, x, ld, y, iy, b, z, iz);
}
static inline void cblas_dsymv(int o, int u, int n, double a, const double *x, int ld, const double *y, int iy, double b, double *z, int iz) {
    typedef void (*function_t)(int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dsymv, function_t)(o, u, n, a, x, ld, y, iy, b, z, iz);
}
static inline void cblas_dger(int o, int m, int n, double a, const double *x, int ix, const double *y, int iy, double *z, int ld) {
    typedef void (*function_t)(int, int, int, double, const double *, int, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dger, function_t)(o, m, n, a, x, ix, y, iy, z, ld);
}
static inline void cblas_dsyr(int o, int u, int n, double a, const double *x, int ix, double *y, int ld) {
    typedef void (*function_t)(int, int, int, double, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dsyr, function_t)(o, u, n, a, x, ix, y, ld);
}
static inline void cblas_dsyr2(int o, int u, int n, double a, const double *x, int ix, const double *y, int iy, double *z, int ld) {
    typedef void (*function_t)(int, int, int, double, const double *, int, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dsyr2, function_t)(o, u, n, a, x, ix, y, iy, z, ld);
}
static inline void cblas_dtrsv(int o, int u, int t, int d, int n, const double *a, int ld, double *x, int ix) {
    typedef void (*function_t)(int, int, int, int, int, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dtrsv, function_t)(o, u, t, d, n, a, ld, x, ix);
}
static inline void cblas_dtrmv(int o, int u, int t, int d, int n, const double *a, int ld, double *x, int ix) {
    typedef void (*function_t)(int, int, int, int, int, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dtrmv, function_t)(o, u, t, d, n, a, ld, x, ix);
}
static inline void cblas_dgemm(int o, int ta, int tb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    typedef void (*function_t)(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dgemm, function_t)(o, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static inline void cblas_dgemmt(int o, int u, int ta, int tb, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    typedef void (*function_t)(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dgemmt, function_t)(o, u, ta, tb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static inline void cblas_dsyrk(int o, int u, int t, int n, int k, double alpha, const double *a, int lda, double beta, double *c, int ldc) {
    typedef void (*function_t)(int, int, int, int, int, double, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dsyrk, function_t)(o, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
static inline void cblas_dsyr2k(int o, int u, int t, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    typedef void (*function_t)(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dsyr2k, function_t)(o, u, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static inline void cblas_dsymm(int o, int s, int u, int m, int n, double alpha, const double *a, int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    typedef void (*function_t)(int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    KOBLAS_RESOLVE(cblas_dsymm, function_t)(o, s, u, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
static inline void cblas_dtrsm(int o, int s, int u, int t, int d, int m, int n, double alpha, const double *a, int lda, double *b, int ldb) {
    typedef void (*function_t)(int, int, int, int, int, int, int, double, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dtrsm, function_t)(o, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
static inline void cblas_dtrmm(int o, int s, int u, int t, int d, int m, int n, double alpha, const double *a, int lda, double *b, int ldb) {
    typedef void (*function_t)(int, int, int, int, int, int, int, double, const double *, int, double *, int);
    KOBLAS_RESOLVE(cblas_dtrmm, function_t)(o, s, u, t, d, m, n, alpha, a, lda, b, ldb);
}
static inline void openblas_set_num_threads(int threads) {
    typedef void (*function_t)(int);
    KOBLAS_RESOLVE(openblas_set_num_threads, function_t)(threads);
}
