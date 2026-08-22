#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "slu_ddefs.h"

extern void dGetDiagU(SuperMatrix *l, double *diagonal);

typedef struct {
    int n;
    SuperMatrix l;
    SuperMatrix u;
    int *perm_c;
    int *perm_r;
    int lnz;
    int unz;
} koblas_superlu_factor;

static int permutation_sign(const int *permutation, int n) {
    int sign = 1;
    unsigned char *seen = calloc((size_t) n, sizeof(*seen));
    if (seen == NULL) return 0;
    for (int start = 0; start < n; ++start) {
        if (seen[start]) continue;
        int length = 0;
        for (int at = start; !seen[at]; at = permutation[at]) {
            seen[at] = 1;
            ++length;
        }
        if ((length & 1) == 0) sign = -sign;
    }
    free(seen);
    return sign;
}

void *koblas_superlu_factorize(
    int n,
    int nnz,
    double *values,
    int *row_idx,
    int *col_ptr,
    int *status,
    int *lnz,
    int *unz
) {
    *status = -1;
    *lnz = 0;
    *unz = 0;
    koblas_superlu_factor *factor = calloc(1, sizeof(*factor));
    if (factor == NULL) return NULL;

    factor->n = n;
    factor->perm_c = intMalloc(n);
    factor->perm_r = intMalloc(n);
    int *etree = intMalloc(n);
    if (factor->perm_c == NULL || factor->perm_r == NULL || etree == NULL) goto cleanup;

    SuperMatrix a;
    SuperMatrix ac;
    dCreate_CompCol_Matrix(&a, n, n, nnz, values, row_idx, col_ptr, SLU_NC, SLU_D, SLU_GE);
    superlu_options_t options;
    set_default_options(&options);
    options.Equil = NO;
    options.IterRefine = NOREFINE;
    options.PrintStat = NO;
    get_perm_c(options.ColPerm, &a, factor->perm_c);
    sp_preorder(&options, &a, factor->perm_c, etree, &ac);

    SuperLUStat_t stat;
    StatInit(&stat);
    int_t info = 0;
    GlobalLU_t glu = {0};
    dgstrf(
        &options,
        &ac,
        sp_ienv(2),
        sp_ienv(1),
        etree,
        NULL,
        0,
        factor->perm_c,
        factor->perm_r,
        &factor->l,
        &factor->u,
        &glu,
        &stat,
        &info
    );
    StatFree(&stat);
    Destroy_CompCol_Permuted(&ac);
    Destroy_SuperMatrix_Store(&a);
    SUPERLU_FREE(etree);
    etree = NULL;
    if (info != 0) {
        *status = info > 0 ? 1 : -1;
        goto cleanup;
    }

    factor->lnz = (int) ((SCformat *) factor->l.Store)->nnz;
    factor->unz = (int) ((NCformat *) factor->u.Store)->nnz;
    *lnz = factor->lnz;
    *unz = factor->unz;
    *status = 0;
    return factor;

cleanup:
    if (etree != NULL) SUPERLU_FREE(etree);
    if (factor->l.Store != NULL) Destroy_SuperNode_Matrix(&factor->l);
    if (factor->u.Store != NULL) Destroy_CompCol_Matrix(&factor->u);
    SUPERLU_FREE(factor->perm_c);
    SUPERLU_FREE(factor->perm_r);
    free(factor);
    return NULL;
}

int koblas_superlu_solve(void *handle, double *rhs, int transpose) {
    koblas_superlu_factor *factor = handle;
    if (factor == NULL) return -1;
    SuperMatrix b;
    dCreate_Dense_Matrix(&b, factor->n, 1, rhs, factor->n, SLU_DN, SLU_D, SLU_GE);
    SuperLUStat_t stat;
    StatInit(&stat);
    int info = 0;
    dgstrs(transpose ? TRANS : NOTRANS, &factor->l, &factor->u, factor->perm_c, factor->perm_r, &b, &stat, &info);
    StatFree(&stat);
    Destroy_SuperMatrix_Store(&b);
    return info;
}

double koblas_superlu_determinant(void *handle) {
    koblas_superlu_factor *factor = handle;
    if (factor == NULL) return NAN;
    double *diagonal = malloc((size_t) factor->n * sizeof(*diagonal));
    if (diagonal == NULL) return NAN;
    dGetDiagU(&factor->l, diagonal);
    double determinant = (double) permutation_sign(factor->perm_r, factor->n) *
        (double) permutation_sign(factor->perm_c, factor->n);
    for (int i = 0; i < factor->n; ++i) determinant *= diagonal[i];
    free(diagonal);
    return determinant;
}

void koblas_superlu_free(void *handle) {
    koblas_superlu_factor *factor = handle;
    if (factor == NULL) return;
    Destroy_SuperNode_Matrix(&factor->l);
    Destroy_CompCol_Matrix(&factor->u);
    SUPERLU_FREE(factor->perm_c);
    SUPERLU_FREE(factor->perm_r);
    free(factor);
}
