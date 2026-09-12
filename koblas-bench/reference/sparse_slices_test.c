#define main vendor_runner_main
#include "vendor_runner.c"
#undef main
#include <float.h>

static void require_test(int condition, const char *message) { if (!condition) fail(message); }
static int same_value(double left, double right) {
    return (isnan(left) && isnan(right)) || (left == right && (left != 0.0 || signbit(left) == signbit(right)));
}

static void check_case(const char *operation, int compact, int shuffled, int exceptional, int library) {
    char text[256];
    int gathers = strstr(operation, "cycle") != NULL || strstr(operation, "gather") != NULL;
    snprintf(text, sizeof(text), "%s+31+sparse-uniform+density=0.3+timing=reuse%s%s", operation,
        gathers && compact ? "+compact=T" : "", shuffled ? "+locality=shuffled" : "");
    bench_case spec = {0}; parse_case(text, 1, &spec);
    scalar_slices = !library;
    work w; setup_work(&w, &spec);
    require_test(w.supported, "expected sparse slice reference support");
    slices_state *s = w.slices;
    int cycle = s->operation == SLICES_CYCLE || s->operation == SLICES_CYCLE_CHECKED;
    if (exceptional) {
        const double values[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, DBL_TRUE_MIN, -DBL_TRUE_MIN, DBL_MAX, -DBL_MAX};
        for (int k = 0; k < s->count; ++k) s->values[k] = values[k % 9];
        if (!cycle) slices_refill(s);
    }
    for (int i = 0; i < s->dimension; ++i) {
        int selected = 0;
        for (int k = 0; k < s->count; ++k) if (s->indices[k] == i) selected = 1;
        if (!selected) { s->accumulator[i] = 19.0; s->marks[i] = 7; }
    }
    for (int k = 0; k < s->count; ++k) { s->out_indices[k] = -1; s->out_values[k] = 23.0; }
    for (int call = 0; call < 3; ++call) {
        double result = invoke_slices(&w);
        int written = 0, expected_status = 0;
        double expected_dot = 0.0;
        for (int k = 0; k < s->count; ++k) {
            double value = s->values[k], expected = value;
            if (cycle) {
                expected = 0.0;
                if (k < (s->count + 1) / 2) {
                    double product = .875 * value;
                    expected = expected + product;
                    if (!isfinite(product) || !isfinite(expected)) expected_status |= 1;
                    if (value != 0.0 && product == 0.0) expected_status |= 2;
                }
                double product = -.875 * value;
                expected = expected + product;
                if (!isfinite(product) || !isfinite(expected)) expected_status |= 1;
                if (value != 0.0 && product == 0.0) expected_status |= 2;
            }
            if (gathers && (!compact || expected != 0.0)) {
                require_test(s->out_indices[written] == s->indices[k], "sparse slice output order differs");
                require_test(same_value(s->out_values[written], expected), "sparse slice output value differs");
                ++written;
            }
            if (cycle || s->operation == SLICES_GATHER_CLEAR || s->operation == SLICES_CLEAR) {
                require_test(same_value(s->accumulator[s->indices[k]], 0.0), "sparse slice scratch not cleared");
                require_test(s->marks[s->indices[k]] == 0, "sparse slice mark not cleared");
            } else {
                require_test(same_value(s->accumulator[s->indices[k]], value), "read only operation changed scratch");
                require_test(s->marks[s->indices[k]] == 1, "read only operation changed mark");
            }
            require_test(s->touched[k] == s->indices[k], "touched order changed");
            if (s->operation == SLICES_DOT || s->operation == SLICES_DOT_CHECKED) {
                double product = value * value, updated = expected_dot + product;
                if (!isfinite(value) || !isfinite(product) || !isfinite(updated)) expected_status |= 1;
                if (isfinite(value) && value != 0.0 && product == 0.0) expected_status |= 2;
                expected_dot = updated;
            }
        }
        if (gathers) {
            require_test(s->written == written, "sparse slice output count differs");
            for (int k = written; k < s->count; ++k) {
                require_test(s->out_indices[k] == -1 && s->out_values[k] == 23.0, "unwritten output tail changed");
            }
        }
        if (s->operation == SLICES_DOT_CHECKED || s->operation == SLICES_CYCLE_CHECKED)
            require_test(s->status == expected_status, "sparse slice diagnostic bits differ");
        if (s->operation == SLICES_DOT_CHECKED)
            require_test(same_value(result, expected_dot + expected_status), "ordered checked reduction differs");
        if (s->operation == SLICES_DOT)
            require_test(same_value(result, expected_dot) || fabs(result - expected_dot) <= 2e-12 * (1 + fabs(expected_dot)), "dot reduction differs");
        for (int i = 0; i < s->dimension; ++i) {
            int selected = 0;
            for (int k = 0; k < s->count; ++k) if (s->indices[k] == i) selected = 1;
            if (!selected) require_test(s->accumulator[i] == 19.0 && s->marks[i] == 7, "untouched scratch changed");
        }
    }
    if (cycle) {
        s->capacity = s->count - 1;
        require_test(!slices_scatter_valid(s, s->count), "insufficient touched capacity accepted");
        s->capacity = s->count;
        int saved = s->indices[s->count - 1];
        s->indices[s->count - 1] = s->dimension;
        require_test(!slices_scatter_valid(s, s->count), "invalid last index accepted");
        s->indices[s->count - 1] = saved;
    }
    if (gathers) {
        s->total = s->count; s->output_capacity = s->count - 1;
        require_test(!slices_gather_valid(s), "insufficient output capacity accepted");
    }
    free_work(&w);
}

int main(void) {
#ifdef USE_MKL
    MKL_Set_Num_Threads(1); MKL_Set_Dynamic(0);
#endif
    scalar_slices = 1;
    char golden_text[] = "sparse-slices-cycle+31+sparse-uniform+density=0.3+timing=reuse+locality=shuffled";
    bench_case golden_case = {0}; parse_case(golden_text, 1, &golden_case);
    work golden_work; setup_work(&golden_work, &golden_case);
    int expected_indices[] = {23, 19, 16, 21, 12, 9, 10, 15, 2};
    require_test(!memcmp(golden_work.slices->indices, expected_indices, sizeof(expected_indices)), "shuffled fixture indices differ");
    require_test(digest(golden_work.slices->values, 9) == UINT64_C(0x82a2f9d0ba820fb8), "shuffled fixture values differ");
    free_work(&golden_work);
    char checked_text[] = "sparse-slices-cycle-checked+3+sparse-uniform+density=1+timing=reuse";
    bench_case checked_case = {0}; parse_case(checked_text, 1, &checked_case);
    work checked_work; setup_work(&checked_work, &checked_case);
    slices_state *checked = checked_work.slices;
    checked->values[0] = DBL_TRUE_MIN;
    slices_scatter(checked, .5, 1, 1);
    require_test(checked->status == 2, "checked scatter missed underflow");
    checked->values[0] = INFINITY;
    slices_scatter(checked, 1.0, 1, 1);
    require_test(checked->status == 3, "checked scatter did not latch diagnostics");
    checked->values[0] = 1e16; checked->values[1] = 1.0; checked->values[2] = -1e16;
    for (int k = 0; k < 3; ++k) checked->accumulator[checked->indices[k]] = 1.0;
    require_test(slices_dot(checked, 1) == 0.0 && checked->status == 0, "checked dot changed reduction order");
    free_work(&checked_work);
    const char *operations[] = {"sparse-slices-cycle", "sparse-slices-cycle-checked", "sparse-slices-gather",
        "sparse-slices-gather-clear", "sparse-slices-clear", "sparse-slices-reduce-dot-checked", "sparse-slices-reduce-dot-unchecked"};
    for (int library = 0; library <= 1; ++library) {
#ifndef USE_MKL
        if (library) continue;
#endif
        for (int operation = 0; operation < 7; ++operation) {
            if (library && (operation == SLICES_CYCLE_CHECKED || operation == SLICES_CLEAR || operation == SLICES_DOT_CHECKED)) continue;
            for (int compact = 0; compact < 2; ++compact) for (int shuffled = 0; shuffled < 2; ++shuffled)
                for (int exceptional = 0; exceptional < 2; ++exceptional)
                    check_case(operations[operation], compact, shuffled, exceptional, library);
        }
    }
    puts("sparse slice reference state and numerical checks passed");
    return 0;
}
