#ifndef KOBLAS_PACKED_TRSM_H
#define KOBLAS_PACKED_TRSM_H

/*
 * Packed right-side triangular solve. T uses the packed-right row groups consumed by the GEMM tile, while
 * X uses its column-major output layout. Exact-zero coefficients are skipped so structural zeroes never
 * form products with non-finite solved values.
 */
KOBLAS_KERNEL void koblas_dense_trsm_tile(
    int32_t valid_rows, int32_t order,
    const double *packed_triangle, int32_t triangle_off,
    int32_t lower, int32_t unit_diag,
    double *x, int32_t x_off
) {
    const double *triangle = packed_triangle + triangle_off;
    double *out = x + x_off;
    if (lower) {
        for (int32_t j = order - 1; j >= 0; j--) {
            double *solved = out + j * KOBLAS_GEMM_TILE;
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < valid_rows; row++) solved[row] /= diagonal;
            }
            for (int32_t column = 0; column < j; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    double *target = out + column * KOBLAS_GEMM_TILE;
                    for (int32_t row = 0; row < valid_rows; row++) {
                        target[row] -= solved[row] * coefficient;
                    }
                }
            }
        }
    } else {
        for (int32_t j = 0; j < order; j++) {
            double *solved = out + j * KOBLAS_GEMM_TILE;
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < valid_rows; row++) solved[row] /= diagonal;
            }
            for (int32_t column = j + 1; column < order; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    double *target = out + column * KOBLAS_GEMM_TILE;
                    for (int32_t row = 0; row < valid_rows; row++) {
                        target[row] -= solved[row] * coefficient;
                    }
                }
            }
        }
    }
}

/*
 * Full four-by-four tiles retain the product and solve intermediates in one local array that an optimizing
 * C compiler scalar-replaces into registers. Edge tiles use the independently vectorizable kernels.
 */
KOBLAS_KERNEL void koblas_dense_gemm_trsm_tile(
    int32_t depth, int32_t valid_rows, int32_t order,
    const double *packed_a, int32_t a_off,
    const double *packed_b, int32_t b_off,
    const double *packed_triangle, int32_t triangle_off,
    int32_t lower, int32_t unit_diag,
    double *x, int32_t x_off
) {
    if (valid_rows != KOBLAS_GEMM_TILE || order != KOBLAS_GEMM_TILE) {
        const double *edge_a = packed_a + a_off;
        const double *edge_b = packed_b + b_off;
        double *edge_x = x + x_off;
        for (int32_t column = 0; column < order; column++) {
            for (int32_t row = 0; row < valid_rows; row++) {
                for (int32_t p = 0; p < depth; p++) {
                    edge_x[column * KOBLAS_GEMM_TILE + row] -=
                        edge_a[p * KOBLAS_GEMM_TILE + row] *
                        edge_b[p * KOBLAS_GEMM_TILE + column];
                }
            }
        }
        koblas_dense_trsm_tile(
            valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off
        );
        return;
    }

    double *out = x + x_off;
    double tile[KOBLAS_GEMM_TILE][KOBLAS_GEMM_TILE];
    for (int32_t column = 0; column < KOBLAS_GEMM_TILE; column++) {
        for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) {
            tile[column][row] = out[column * KOBLAS_GEMM_TILE + row];
        }
    }
    const double *ap = packed_a + a_off;
    const double *bp = packed_b + b_off;
    for (int32_t p = 0; p < depth; p++) {
        for (int32_t column = 0; column < KOBLAS_GEMM_TILE; column++) {
            const double coefficient = bp[column];
            for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) {
                tile[column][row] -= ap[row] * coefficient;
            }
        }
        ap += KOBLAS_GEMM_TILE;
        bp += KOBLAS_GEMM_TILE;
    }
    const double *triangle = packed_triangle + triangle_off;
    if (lower) {
        for (int32_t j = KOBLAS_GEMM_TILE - 1; j >= 0; j--) {
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) tile[j][row] /= diagonal;
            }
            for (int32_t column = 0; column < j; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) {
                        tile[column][row] -= tile[j][row] * coefficient;
                    }
                }
            }
        }
    } else {
        for (int32_t j = 0; j < KOBLAS_GEMM_TILE; j++) {
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) tile[j][row] /= diagonal;
            }
            for (int32_t column = j + 1; column < KOBLAS_GEMM_TILE; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) {
                        tile[column][row] -= tile[j][row] * coefficient;
                    }
                }
            }
        }
    }
    for (int32_t column = 0; column < KOBLAS_GEMM_TILE; column++) {
        for (int32_t row = 0; row < KOBLAS_GEMM_TILE; row++) {
            out[column * KOBLAS_GEMM_TILE + row] = tile[column][row];
        }
    }
}

#endif
