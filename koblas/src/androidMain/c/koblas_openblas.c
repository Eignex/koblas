/*
 * The JNI entry points behind com.eignex.koblas.vendor.AndroidCblas.
 *
 * ART has no foreign function interface, so a Kotlin call reaches CBLAS through these: each one pins its
 * arrays, offsets the vector operands, and makes exactly one CBLAS call with the caller's own arguments.
 * OpenBLAS is linked in statically and its symbols are hidden, so an application that loads its own BLAS
 * cannot bind this library's calls to that one or the other way round.
 *
 * The prototypes are declared here rather than taken from OpenBLAS's header because the ABI is fixed: LP64
 * integers, double precision and the unsuffixed CBLAS names, with the enumerations passed as the integers the
 * standard gives them.
 */
#include <dlfcn.h>
#include <jni.h>
#include <string.h>

typedef int blasint;

double cblas_ddot(blasint, const double *, blasint, const double *, blasint);
double cblas_dnrm2(blasint, const double *, blasint);
double cblas_dasum(blasint, const double *, blasint);
size_t cblas_idamax(blasint, const double *, blasint);
void cblas_daxpy(blasint, double, const double *, blasint, double *, blasint);
void cblas_dscal(blasint, double, double *, blasint);
void cblas_dcopy(blasint, const double *, blasint, double *, blasint);
void cblas_dswap(blasint, double *, blasint, double *, blasint);
void cblas_drot(blasint, double *, blasint, double *, blasint, double, double);
void cblas_dgemv(int, int, blasint, blasint, double, const double *, blasint, const double *, blasint, double,
                 double *, blasint);
void cblas_dsymv(int, int, blasint, double, const double *, blasint, const double *, blasint, double, double *,
                 blasint);
void cblas_dger(int, blasint, blasint, double, const double *, blasint, const double *, blasint, double *,
                blasint);
void cblas_dsyr(int, int, blasint, double, const double *, blasint, double *, blasint);
void cblas_dsyr2(int, int, blasint, double, const double *, blasint, const double *, blasint, double *, blasint);
void cblas_dtrsv(int, int, int, int, blasint, const double *, blasint, double *, blasint);
void cblas_dtrmv(int, int, int, int, blasint, const double *, blasint, double *, blasint);
void cblas_dgemm(int, int, int, blasint, blasint, blasint, double, const double *, blasint, const double *,
                 blasint, double, double *, blasint);
void cblas_dsymm(int, int, int, blasint, blasint, double, const double *, blasint, const double *, blasint,
                 double, double *, blasint);
void cblas_dsyrk(int, int, int, blasint, blasint, double, const double *, blasint, double, double *, blasint);
void cblas_dsyr2k(int, int, int, blasint, blasint, double, const double *, blasint, const double *, blasint,
                  double, double *, blasint);
void cblas_dtrmm(int, int, int, int, int, blasint, blasint, double, const double *, blasint, double *, blasint);
void cblas_dtrsm(int, int, int, int, int, blasint, blasint, double, const double *, blasint, double *, blasint);
void cblas_dgemmt(int, int, int, int, blasint, blasint, double, const double *, blasint, const double *,
                  blasint, double, double *, blasint);
char *openblas_get_config(void);
char *openblas_get_corename(void);
int openblas_get_num_threads(void);
void openblas_set_num_threads(int);

#define COL_MAJOR 102
#define MAX_PINS 3

/*
 * The arrays one call holds, pinned for as long as the call runs and no longer.
 *
 * Two operands may be one array, as a triangular solve's matrix and right-hand side can be, so an array is
 * pinned once however many operands name it; pinning it twice would, on a runtime that copies, write back
 * whichever copy is released last. Whether an array is written back is decided per array for the same
 * reason: one written through any of its operands is released with its changes, and one only read is
 * released with JNI_ABORT, so a runtime that copied it does not write stale values over the caller's.
 */
typedef struct {
    int count;
    jdoubleArray arrays[MAX_PINS];
    double *bases[MAX_PINS];
    int owner[MAX_PINS];
    int written[MAX_PINS];
} pins;

/* Resolves aliases before any array is pinned, because no JNI call is allowed while one is. */
static void pins_add(JNIEnv *env, pins *p, jdoubleArray array, int written) {
    int i = p->count++;
    p->arrays[i] = array;
    p->owner[i] = i;
    p->written[i] = written;
    for (int j = 0; j < i; j++) {
        if (p->owner[j] == j && (*env)->IsSameObject(env, array, p->arrays[j])) {
            p->owner[i] = j;
            p->written[j] |= written;
            break;
        }
    }
}

static void pins_release(JNIEnv *env, pins *p, int upto) {
    for (int i = upto - 1; i >= 0; i--) {
        if (p->owner[i] != i || p->bases[i] == NULL) continue;
        (*env)->ReleasePrimitiveArrayCritical(env, p->arrays[i], p->bases[i], p->written[i] ? 0 : JNI_ABORT);
    }
}

/* Pins every array added, or none: on failure the runtime has already raised OutOfMemoryError. */
static int pins_acquire(JNIEnv *env, pins *p) {
    for (int i = 0; i < p->count; i++) {
        if (p->owner[i] != i) {
            p->bases[i] = p->bases[p->owner[i]];
            continue;
        }
        p->bases[i] = (*env)->GetPrimitiveArrayCritical(env, p->arrays[i], NULL);
        if (p->bases[i] == NULL) {
            pins_release(env, p, i);
            return 0;
        }
    }
    return 1;
}

#define JNI_FN(type, name) JNIEXPORT type JNICALL Java_com_eignex_koblas_vendor_AndroidCblas_##name

/* Level 1. Vector operands carry an offset because a vector is a window onto a shared array. */

JNI_FN(jdouble, ddot)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX, jdoubleArray y,
                      jint yOff, jint incY) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 0);
    if (!pins_acquire(env, &p)) return 0.0;
    double result = cblas_ddot(n, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY);
    pins_release(env, &p, p.count);
    return result;
}

JNI_FN(jdouble, dnrm2)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    if (!pins_acquire(env, &p)) return 0.0;
    double result = cblas_dnrm2(n, p.bases[0] + xOff, incX);
    pins_release(env, &p, p.count);
    return result;
}

JNI_FN(jdouble, dasum)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    if (!pins_acquire(env, &p)) return 0.0;
    double result = cblas_dasum(n, p.bases[0] + xOff, incX);
    pins_release(env, &p, p.count);
    return result;
}

JNI_FN(jint, idamax)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    if (!pins_acquire(env, &p)) return 0;
    jint result = (jint) cblas_idamax(n, p.bases[0] + xOff, incX);
    pins_release(env, &p, p.count);
    return result;
}

JNI_FN(void, daxpy)(JNIEnv *env, jclass cls, jint n, jdouble alpha, jdoubleArray x, jint xOff, jint incX,
                    jdoubleArray y, jint yOff, jint incY) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_daxpy(n, alpha, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dscal)(JNIEnv *env, jclass cls, jint n, jdouble alpha, jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, x, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dscal(n, alpha, p.bases[0] + xOff, incX);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dcopy)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX, jdoubleArray y,
                    jint yOff, jint incY) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dcopy(n, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dswap)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX, jdoubleArray y,
                    jint yOff, jint incY) {
    pins p = {0};
    pins_add(env, &p, x, 1);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dswap(n, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY);
    pins_release(env, &p, p.count);
}

JNI_FN(void, drot)(JNIEnv *env, jclass cls, jint n, jdoubleArray x, jint xOff, jint incX, jdoubleArray y,
                   jint yOff, jint incY, jdouble c, jdouble s) {
    pins p = {0};
    pins_add(env, &p, x, 1);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_drot(n, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY, c, s);
    pins_release(env, &p, p.count);
}

/* Level 2. A matrix is always the whole of its array, so it carries a leading dimension and no offset. */

JNI_FN(void, dgemv)(JNIEnv *env, jclass cls, jint trans, jint m, jint n, jdouble alpha, jdoubleArray a,
                    jint lda, jdoubleArray x, jint xOff, jint incX, jdouble beta, jdoubleArray y, jint yOff,
                    jint incY) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dgemv(COL_MAJOR, trans, m, n, alpha, p.bases[0], lda, p.bases[1] + xOff, incX, beta,
                p.bases[2] + yOff, incY);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsymv)(JNIEnv *env, jclass cls, jint uplo, jint n, jdouble alpha, jdoubleArray a, jint lda,
                    jdoubleArray x, jint xOff, jint incX, jdouble beta, jdoubleArray y, jint yOff, jint incY) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsymv(COL_MAJOR, uplo, n, alpha, p.bases[0], lda, p.bases[1] + xOff, incX, beta, p.bases[2] + yOff,
                incY);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dger)(JNIEnv *env, jclass cls, jint m, jint n, jdouble alpha, jdoubleArray x, jint xOff, jint incX,
                   jdoubleArray y, jint yOff, jint incY, jdoubleArray a, jint lda) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 0);
    pins_add(env, &p, a, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dger(COL_MAJOR, m, n, alpha, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY, p.bases[2], lda);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsyr)(JNIEnv *env, jclass cls, jint uplo, jint n, jdouble alpha, jdoubleArray x, jint xOff,
                   jint incX, jdoubleArray a, jint lda) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, a, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsyr(COL_MAJOR, uplo, n, alpha, p.bases[0] + xOff, incX, p.bases[1], lda);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsyr2)(JNIEnv *env, jclass cls, jint uplo, jint n, jdouble alpha, jdoubleArray x, jint xOff,
                    jint incX, jdoubleArray y, jint yOff, jint incY, jdoubleArray a, jint lda) {
    pins p = {0};
    pins_add(env, &p, x, 0);
    pins_add(env, &p, y, 0);
    pins_add(env, &p, a, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsyr2(COL_MAJOR, uplo, n, alpha, p.bases[0] + xOff, incX, p.bases[1] + yOff, incY, p.bases[2], lda);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dtrsv)(JNIEnv *env, jclass cls, jint uplo, jint trans, jint diag, jint n, jdoubleArray a, jint lda,
                    jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, x, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dtrsv(COL_MAJOR, uplo, trans, diag, n, p.bases[0], lda, p.bases[1] + xOff, incX);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dtrmv)(JNIEnv *env, jclass cls, jint uplo, jint trans, jint diag, jint n, jdoubleArray a, jint lda,
                    jdoubleArray x, jint xOff, jint incX) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, x, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dtrmv(COL_MAJOR, uplo, trans, diag, n, p.bases[0], lda, p.bases[1] + xOff, incX);
    pins_release(env, &p, p.count);
}

/* Level 3. */

JNI_FN(void, dgemm)(JNIEnv *env, jclass cls, jint transA, jint transB, jint m, jint n, jint k, jdouble alpha,
                    jdoubleArray a, jint lda, jdoubleArray b, jint ldb, jdouble beta, jdoubleArray c, jint ldc) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 0);
    pins_add(env, &p, c, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dgemm(COL_MAJOR, transA, transB, m, n, k, alpha, p.bases[0], lda, p.bases[1], ldb, beta, p.bases[2],
                ldc);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsymm)(JNIEnv *env, jclass cls, jint side, jint uplo, jint m, jint n, jdouble alpha,
                    jdoubleArray a, jint lda, jdoubleArray b, jint ldb, jdouble beta, jdoubleArray c, jint ldc) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 0);
    pins_add(env, &p, c, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsymm(COL_MAJOR, side, uplo, m, n, alpha, p.bases[0], lda, p.bases[1], ldb, beta, p.bases[2], ldc);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsyrk)(JNIEnv *env, jclass cls, jint uplo, jint trans, jint n, jint k, jdouble alpha,
                    jdoubleArray a, jint lda, jdouble beta, jdoubleArray c, jint ldc) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, c, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsyrk(COL_MAJOR, uplo, trans, n, k, alpha, p.bases[0], lda, beta, p.bases[1], ldc);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dsyr2k)(JNIEnv *env, jclass cls, jint uplo, jint trans, jint n, jint k, jdouble alpha,
                     jdoubleArray a, jint lda, jdoubleArray b, jint ldb, jdouble beta, jdoubleArray c,
                     jint ldc) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 0);
    pins_add(env, &p, c, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dsyr2k(COL_MAJOR, uplo, trans, n, k, alpha, p.bases[0], lda, p.bases[1], ldb, beta, p.bases[2], ldc);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dtrmm)(JNIEnv *env, jclass cls, jint side, jint uplo, jint trans, jint diag, jint m, jint n,
                    jdouble alpha, jdoubleArray a, jint lda, jdoubleArray b, jint ldb) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dtrmm(COL_MAJOR, side, uplo, trans, diag, m, n, alpha, p.bases[0], lda, p.bases[1], ldb);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dtrsm)(JNIEnv *env, jclass cls, jint side, jint uplo, jint trans, jint diag, jint m, jint n,
                    jdouble alpha, jdoubleArray a, jint lda, jdoubleArray b, jint ldb) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dtrsm(COL_MAJOR, side, uplo, trans, diag, m, n, alpha, p.bases[0], lda, p.bases[1], ldb);
    pins_release(env, &p, p.count);
}

JNI_FN(void, dgemmt)(JNIEnv *env, jclass cls, jint uplo, jint transA, jint transB, jint n, jint k,
                     jdouble alpha, jdoubleArray a, jint lda, jdoubleArray b, jint ldb, jdouble beta,
                     jdoubleArray c, jint ldc) {
    pins p = {0};
    pins_add(env, &p, a, 0);
    pins_add(env, &p, b, 0);
    pins_add(env, &p, c, 1);
    if (!pins_acquire(env, &p)) return;
    cblas_dgemmt(COL_MAJOR, uplo, transA, transB, n, k, alpha, p.bases[0], lda, p.bases[1], ldb, beta,
                 p.bases[2], ldc);
    pins_release(env, &p, p.count);
}

/* The library's own description of itself, which is what the ABI check reads. */

JNI_FN(jstring, config)(JNIEnv *env, jclass cls) {
    const char *config = openblas_get_config();
    return (*env)->NewStringUTF(env, config != NULL ? config : "");
}

/* The core whose kernels the dispatch chose at load, which the build's configuration string does not say. */
JNI_FN(jstring, coreName)(JNIEnv *env, jclass cls) {
    const char *name = openblas_get_corename();
    return (*env)->NewStringUTF(env, name != NULL ? name : "");
}

JNI_FN(jint, threads)(JNIEnv *env, jclass cls) { return openblas_get_num_threads(); }

JNI_FN(void, holdToOneThread)(JNIEnv *env, jclass cls) { openblas_set_num_threads(1); }

/* The file this code was loaded from, asked of the loader rather than assumed from the name it was asked for. */
JNI_FN(jstring, libraryPath)(JNIEnv *env, jclass cls) {
    Dl_info info;
    memset(&info, 0, sizeof(info));
    if (dladdr((void *) &openblas_get_config, &info) == 0 || info.dli_fname == NULL) return NULL;
    return (*env)->NewStringUTF(env, info.dli_fname);
}
