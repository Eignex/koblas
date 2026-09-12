/* Benchmark-only compositions over persistent caller-owned buffers. */
static int scalar_slices;

enum { SLICES_CYCLE, SLICES_CYCLE_CHECKED, SLICES_GATHER, SLICES_GATHER_CLEAR,
       SLICES_CLEAR, SLICES_DOT_CHECKED, SLICES_DOT };
struct slices_state {
    int dimension, count, capacity, output_capacity, operation, compact, use_mkl;
    int total, written, status;
    int *indices, *marks, *touched, *out_indices;
    double *values, *accumulator, *out_values, *scratch;
};

static int slices_indices_valid(const slices_state *s, const int *indices, int count) {
    if (count < 0 || count > s->count) return 0;
    for (int k = 0; k < count; ++k) if (indices[k] < 0 || indices[k] >= s->dimension) return 0;
    return 1;
}

static int slices_scatter_valid(const slices_state *s, int count) {
    if (s->values == s->accumulator || s->indices == s->marks || s->marks == s->touched ||
        s->indices == s->touched || s->total < 0 || s->total > s->capacity ||
        !slices_indices_valid(s, s->indices, count)) return 0;
    int fresh = 0;
    for (int k = 0; k < count; ++k) if (s->marks[s->indices[k]] != 1) ++fresh;
    return fresh <= s->capacity - s->total;
}

static void slices_scatter(slices_state *s, double alpha, int count, int checked) {
    if (!slices_scatter_valid(s, count)) fail("sparse slices scatter validation failed");
#ifdef USE_MKL
    if (s->use_mkl) {
        /* Multiplication is rounded before accumulation, even if the vendor AXPY uses FMA. */
        for (int k = 0; k < count; ++k) {
            int index = s->indices[k];
            if (s->marks[index] != 1) {
                s->accumulator[index] = 0.0;
                s->marks[index] = 1;
                s->touched[s->total++] = index;
            }
            s->scratch[k] = alpha * s->values[k];
        }
        cblas_daxpyi(count, 1.0, s->scratch, s->indices, s->accumulator);
        return;
    }
#endif
    for (int k = 0; k < count; ++k) {
        int index = s->indices[k];
        if (s->marks[index] != 1) {
            s->accumulator[index] = 0.0;
            s->marks[index] = 1;
            s->touched[s->total++] = index;
        }
        double value = s->values[k], product = alpha * value;
        double updated = s->accumulator[index] + product;
        if (checked) {
            if (!isfinite(product) || !isfinite(updated)) s->status |= 1;
            if (alpha != 0.0 && value != 0.0 && product == 0.0) s->status |= 2;
        }
        s->accumulator[index] = updated;
    }
}

static int slices_gather_valid(const slices_state *s) {
    return s->total <= s->output_capacity && s->out_values != s->accumulator &&
        s->out_indices != s->touched && s->marks != s->touched && s->marks != s->out_indices &&
        slices_indices_valid(s, s->touched, s->total);
}

static void slices_gather(slices_state *s, int clear) {
    if (!slices_gather_valid(s)) fail("sparse slices gather validation failed");
    s->written = 0;
#ifdef USE_MKL
    if (s->use_mkl) {
        /* Compaction must leave the unwritten output tail unchanged. */
        double *gathered = s->compact ? s->scratch : s->out_values;
        if (clear) cblas_dgthrz(s->total, s->accumulator, gathered, s->touched);
        else cblas_dgthr(s->total, s->accumulator, gathered, s->touched);
        for (int k = 0; k < s->total; ++k) {
            int index = s->touched[k];
            double value = gathered[k];
            if (!s->compact || value != 0.0) {
                s->out_indices[s->written] = index;
                if (s->compact) s->out_values[s->written] = value;
                ++s->written;
            }
            if (clear) s->marks[index] = 0;
        }
        return;
    }
#endif
    for (int k = 0; k < s->total; ++k) {
        int index = s->touched[k];
        double value = s->accumulator[index];
        if (!s->compact || value != 0.0) {
            s->out_indices[s->written] = index;
            s->out_values[s->written++] = value;
        }
        if (clear) { s->accumulator[index] = 0.0; s->marks[index] = 0; }
    }
}

static void slices_refill(slices_state *s) {
    for (int k = 0; k < s->count; ++k) {
        s->accumulator[s->touched[k]] = s->values[k];
        s->marks[s->touched[k]] = 1;
    }
}

static double slices_dot(slices_state *s, int checked) {
    if (!slices_indices_valid(s, s->indices, s->count)) fail("sparse slices dot validation failed");
    s->status = 0;
#ifdef USE_MKL
    if (s->use_mkl) return cblas_ddoti(s->count, s->values, s->indices, s->accumulator);
#endif
    double result = 0.0;
    for (int k = 0; k < s->count; ++k) {
        double left = s->values[k], right = s->accumulator[s->indices[k]];
        double product = left * right, updated = result + product;
        if (checked) {
            if (!isfinite(left) || !isfinite(right) || !isfinite(product) || !isfinite(updated)) s->status |= 1;
            if (isfinite(left) && isfinite(right) && left != 0.0 && right != 0.0 && product == 0.0) s->status |= 2;
        }
        result = updated;
    }
    return result + (checked ? s->status : 0);
}

static double invoke_slices(work *w) {
    slices_state *s = w->slices;
    switch (s->operation) {
    case SLICES_CYCLE: case SLICES_CYCLE_CHECKED:
        s->status = 0;
        slices_scatter(s, .875, (s->count + 1) / 2, s->operation == SLICES_CYCLE_CHECKED);
        slices_scatter(s, -.875, s->count, s->operation == SLICES_CYCLE_CHECKED);
        slices_gather(s, 1);
        s->total = 0;
        break;
    case SLICES_GATHER: slices_gather(s, 0); break;
    case SLICES_GATHER_CLEAR: slices_refill(s); slices_gather(s, 1); break;
    case SLICES_CLEAR:
        slices_refill(s);
        if (s->marks == s->touched || !slices_indices_valid(s, s->touched, s->count)) fail("sparse slices clear validation failed");
        for (int k = 0; k < s->count; ++k) { s->accumulator[s->touched[k]] = 0.0; s->marks[s->touched[k]] = 0; }
        return s->accumulator[s->touched[0]] + s->accumulator[s->touched[s->count - 1]] +
            s->marks[s->touched[0]] + s->marks[s->touched[s->count - 1]];
    case SLICES_DOT_CHECKED: return slices_dot(s, 1);
    case SLICES_DOT: return slices_dot(s, 0);
    }
    return s->written + s->status + (s->written == 0 ? 0.0 :
        s->out_values[0] + s->out_values[s->written - 1] + s->out_indices[0] + s->out_indices[s->written - 1]);
}

static const char *slices_timing(int operation) {
    static const char *timings[] = { "slices-cycle-v1", "slices-cycle-checked-v1", "slices-gather-v1",
        "slices-refill-gather-clear-v1", "slices-refill-clear-v1", "slices-ordered-dot-checked-v1", "slices-dot-v1" };
    return timings[operation];
}

static void setup_slices(work *w) {
    bench_case *spec = w->spec;
    w->comparison = "unsupported"; w->timing = "sparse-slices";
    if (strcmp(option(spec, "timing", ""), "reuse")) return;
    const char *operations[] = { "sparse-slices-cycle", "sparse-slices-cycle-checked", "sparse-slices-gather",
        "sparse-slices-gather-clear", "sparse-slices-clear", "sparse-slices-reduce-dot-checked", "sparse-slices-reduce-dot-unchecked" };
    int operation = -1;
    for (int i = 0; i < 7; ++i) if (!strcmp(spec->operation, operations[i])) operation = i;
    if (operation < 0) fail("unknown sparse slices comparison");
    w->timing = slices_timing(operation);
    if (!scalar_slices) {
#ifndef USE_MKL
        return;
#else
        if (operation == SLICES_CYCLE_CHECKED || operation == SLICES_CLEAR || operation == SLICES_DOT_CHECKED) return;
#endif
    }
    slices_state *s = allocate(1, sizeof(*s));
    w->slices = s;
    sparse_fixture fixture = make_sparse(spec->dims[0], 1, strtod(option(spec, "density", ".01"), NULL), 1, 0, 1);
    s->dimension = spec->dims[0]; s->count = fixture.nnz; s->capacity = s->count; s->output_capacity = s->count;
    s->operation = operation; s->compact = !strcmp(option(spec, "compact", "N"), "T"); s->use_mkl = !scalar_slices;
    s->indices = fixture.row_idx; s->values = fixture.values; free(fixture.col_ptr);
    s->marks = allocate(s->dimension, sizeof(int)); s->touched = allocate(s->count, sizeof(int));
    s->out_indices = allocate(s->count, sizeof(int)); s->accumulator = allocate(s->dimension, sizeof(double));
    s->out_values = allocate(s->count, sizeof(double)); s->scratch = allocate(s->count, sizeof(double));
    if (!strcmp(option(spec, "locality", "sorted"), "shuffled")) {
        portable_random random = stream(71);
        for (int i = s->count - 1; i > 0; --i) {
            int j = (int)((random_long(&random) >> 1) % (uint64_t)(i + 1));
            int index = s->indices[i]; s->indices[i] = s->indices[j]; s->indices[j] = index;
            double value = s->values[i]; s->values[i] = s->values[j]; s->values[j] = value;
        }
    }
    memcpy(s->touched, s->indices, (size_t)s->count * sizeof(int));
    if (operation != SLICES_CYCLE && operation != SLICES_CYCLE_CHECKED) { s->total = s->count; slices_refill(s); }
    w->supported = 1; w->comparison = "composed"; w->invoke = invoke_slices;
    w->actual_kernel = scalar_slices ? "scalar-c-validated-slices" : "onemkl-with-validated-bookkeeping";
}

static void free_slices(work *w) {
    slices_state *s = w->slices;
    if (!s) return;
    free(s->indices); free(s->values); free(s->marks); free(s->touched); free(s->out_indices);
    free(s->accumulator); free(s->out_values); free(s->scratch); free(s);
}
