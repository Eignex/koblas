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
 * Full four-by-four tiles retain the product intermediates in scalar locals. Edge tiles use loops over only
 * their logical rows and columns so callers can provide exact-size buffers without padding writes.
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
    double c00 = out[0], c10 = out[1], c20 = out[2], c30 = out[3];
    double c01 = out[4], c11 = out[5], c21 = out[6], c31 = out[7];
    double c02 = out[8], c12 = out[9], c22 = out[10], c32 = out[11];
    double c03 = out[12], c13 = out[13], c23 = out[14], c33 = out[15];
    const double *ap = packed_a + a_off;
    const double *bp = packed_b + b_off;
    for (int32_t p = 0; p < depth; p++) {
        const double a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
        double coefficient = bp[0];
        c00 -= a0 * coefficient; c10 -= a1 * coefficient;
        c20 -= a2 * coefficient; c30 -= a3 * coefficient;
        coefficient = bp[1];
        c01 -= a0 * coefficient; c11 -= a1 * coefficient;
        c21 -= a2 * coefficient; c31 -= a3 * coefficient;
        coefficient = bp[2];
        c02 -= a0 * coefficient; c12 -= a1 * coefficient;
        c22 -= a2 * coefficient; c32 -= a3 * coefficient;
        coefficient = bp[3];
        c03 -= a0 * coefficient; c13 -= a1 * coefficient;
        c23 -= a2 * coefficient; c33 -= a3 * coefficient;
        ap += KOBLAS_GEMM_TILE;
        bp += KOBLAS_GEMM_TILE;
    }
    const double *triangle = packed_triangle + triangle_off;
#define KOBLAS_DIVIDE_COLUMN(a, b, c, d, diagonal) \
    do { a /= diagonal; b /= diagonal; c /= diagonal; d /= diagonal; } while (0)
#define KOBLAS_SUBTRACT_COLUMN(a, b, c, d, sa, sb, sc, sd, coefficient) \
    do { \
        a -= sa * coefficient; b -= sb * coefficient; \
        c -= sc * coefficient; d -= sd * coefficient; \
    } while (0)
    if (lower) {
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c03, c13, c23, c33, triangle[15]);
        double coefficient = triangle[12];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c03, c13, c23, c33, coefficient);
        coefficient = triangle[13];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c03, c13, c23, c33, coefficient);
        coefficient = triangle[14];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c03, c13, c23, c33, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c02, c12, c22, c32, triangle[10]);
        coefficient = triangle[8];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c02, c12, c22, c32, coefficient);
        coefficient = triangle[9];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c02, c12, c22, c32, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c01, c11, c21, c31, triangle[5]);
        coefficient = triangle[4];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c01, c11, c21, c31, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c00, c10, c20, c30, triangle[0]);
    } else {
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c00, c10, c20, c30, triangle[0]);
        double coefficient = triangle[1];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c00, c10, c20, c30, coefficient);
        coefficient = triangle[2];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c00, c10, c20, c30, coefficient);
        coefficient = triangle[3];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c00, c10, c20, c30, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c01, c11, c21, c31, triangle[5]);
        coefficient = triangle[6];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c01, c11, c21, c31, coefficient);
        coefficient = triangle[7];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c01, c11, c21, c31, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c02, c12, c22, c32, triangle[10]);
        coefficient = triangle[11];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c02, c12, c22, c32, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c03, c13, c23, c33, triangle[15]);
    }
#undef KOBLAS_SUBTRACT_COLUMN
#undef KOBLAS_DIVIDE_COLUMN
    out[0] = c00; out[1] = c10; out[2] = c20; out[3] = c30;
    out[4] = c01; out[5] = c11; out[6] = c21; out[7] = c31;
    out[8] = c02; out[9] = c12; out[10] = c22; out[11] = c32;
    out[12] = c03; out[13] = c13; out[14] = c23; out[15] = c33;
}

#endif
