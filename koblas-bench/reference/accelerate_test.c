#define main vendor_main
#include "vendor_runner.c"
#undef main

static void assert_close(double actual, double expected) {
    if (!isfinite(actual) || fabs(actual - expected) > 2e-12 * (1 + fabs(expected))) {
        fprintf(stderr, "actual %.17g expected %.17g\n", actual, expected);
        fail("Accelerate disagrees with scalar reference");
    }
}

static void check_case(const char *id) {
    char line[2048];
    snprintf(line, sizeof(line), "%s", id);
    bench_case spec = {0};
    if (!parse_case(line, 1, &spec)) fail("missing test case");
    work w;
    setup_work(&w, &spec);
    if (!w.supported) fail("missing Accelerate operation");
    int rows = spec.dims[0];
    double expected[128] = {0};
    double reference = 0;
    if (!strcmp(spec.operation, "spdot")) {
        double dense[128];
        fill_vector(dense, rows, 2);
        for (int p = 0; p < w.sa.nnz; ++p) reference += w.sa.values[p] * dense[w.sa.row_idx[p]];
    } else if (!strcmp(spec.operation, "spnrm2") || !strcmp(spec.operation, "spasum")) {
        for (int p = 0; p < w.sa.nnz; ++p) {
            double value = w.sa.values[p];
            reference += !strcmp(spec.operation, "spnrm2") ? value * value : fabs(value);
        }
        if (!strcmp(spec.operation, "spnrm2")) reference = sqrt(reference);
    } else if (!strcmp(spec.operation, "spaxpy")) {
        copy_values(expected, w.initial, rows);
        for (int p = 0; p < w.sa.nnz; ++p) expected[w.sa.row_idx[p]] += .875 * w.sa.values[p];
    } else {
        int rhs = !strcmp(spec.operation, "spmm") ? spec.dims[1] : 1;
        for (int i = 0; i < rows * rhs; ++i) expected[i] = -.25 * w.initial[i];
        for (int j = 0; j < rhs; ++j) for (int column = 0; column < w.sa.cols; ++column) {
            double x = !strcmp(spec.operation, "spmm") ? w.b[column + j * w.sa.cols] : w.x[column];
            for (int p = w.sa.col_ptr[column]; p < w.sa.col_ptr[column + 1]; ++p) {
                expected[w.sa.row_idx[p] + j * rows] += .875 * w.sa.values[p] * x;
            }
        }
    }
    // A second call also verifies reset semantics for mutable outputs and prepared handles.
    for (int repeat = 0; repeat < 2; ++repeat) {
        double actual = w.invoke(&w);
        if (!strcmp(spec.operation, "spdot") || !strcmp(spec.operation, "spnrm2") || !strcmp(spec.operation, "spasum")) {
            assert_close(actual, reference);
        } else {
            int count = !strcmp(spec.operation, "spmm") ? rows * spec.dims[1] : rows;
            double *output = !strcmp(spec.operation, "spmm") ? w.c : w.y;
            for (int i = 0; i < count; ++i) assert_close(output[i], expected[i]);
        }
    }
    free_work(&w);
}

int main(void) {
    check_case("spdot+19+sparse-uniform+density=0.3");
    check_case("spaxpy+19+sparse-uniform+density=0.3");
    check_case("spnrm2+19+sparse-uniform+density=0.3");
    check_case("spasum+19+sparse-uniform+density=0.3");
    check_case("spgemv+11x7+sparse-uniform+density=0.3+mode=prepared");
    check_case("spgemv+11x7+sparse-uniform+density=0.3+mode=oneshot");
    check_case("spmm+11x3x7+sparse-uniform+density=0.3+mode=prepared");
    check_case("spmm+11x3x7+sparse-uniform+density=0.3+mode=oneshot");
    puts("Accelerate sparse numerical checks passed");
    return 0;
}
