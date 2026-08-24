#include "basiclu.h"

#include <stdint.h>
#include <stdlib.h>

struct koblas_basiclu {
    struct basiclu_object object;
};

struct koblas_basiclu *koblas_basiclu_create(int64_t dimension) {
    struct koblas_basiclu *handle = calloc(1, sizeof(*handle));
    if (handle == NULL) return NULL;
    if (basiclu_obj_initialize(&handle->object, dimension) != BASICLU_OK) {
        basiclu_obj_free(&handle->object);
        free(handle);
        return NULL;
    }
    return handle;
}

void koblas_basiclu_free(struct koblas_basiclu *handle) {
    if (handle == NULL) return;
    basiclu_obj_free(&handle->object);
    free(handle);
}

int64_t koblas_basiclu_factorize(
    struct koblas_basiclu *handle,
    const int64_t *begin,
    const int64_t *end,
    const int64_t *row,
    const double *value
) {
    return basiclu_obj_factorize(&handle->object, begin, end, row, value);
}

int64_t koblas_basiclu_solve(
    struct koblas_basiclu *handle,
    const double *rhs,
    double *lhs,
    int transpose
) {
    return basiclu_obj_solve_dense(&handle->object, rhs, lhs, transpose ? 't' : 'n');
}

int64_t koblas_basiclu_prepare_update(
    struct koblas_basiclu *handle,
    int64_t nz,
    const int64_t *index,
    const double *value,
    int64_t leaving_column
) {
    int64_t status = basiclu_obj_solve_for_update(&handle->object, nz, index, value, 'n', 0);
    if (status != BASICLU_OK) return status;
    return basiclu_obj_solve_for_update(&handle->object, 1, &leaving_column, NULL, 't', 0);
}

int64_t koblas_basiclu_update(struct koblas_basiclu *handle) {
    return basiclu_obj_update(&handle->object, 0.0);
}

double koblas_basiclu_info(const struct koblas_basiclu *handle, int64_t index) {
    return handle->object.xstore[index];
}
