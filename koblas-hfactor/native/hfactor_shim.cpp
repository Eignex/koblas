/*
 * The C entry points koblas binds to. HFactor is a C++ class with no C API, so the seam is drawn here:
 * one handle owns the factorization, the constraint matrix it draws basis columns from, and the two
 * vectors an update reads.
 *
 * Indices are int32 throughout, matching HighsInt in this build and koblas's own CSC arrays. Vectors cross
 * packed, as a count with parallel index and value arrays, so a solve whose result is sparse costs its own
 * nonzeros to carry rather than the dimension.
 */
#include "util/HFactor.h"
#include "util/HVector.h"

#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

namespace {

struct Handle {
    HFactor factor;
    HighsInt num_row = 0;
    HighsInt num_col = 0;
    std::vector<HighsInt> start;
    std::vector<HighsInt> index;
    std::vector<double> value;
    // HFactor's own slot order, which build permutes away from the caller's.
    std::vector<HighsInt> basic_index;
    std::vector<HighsInt> to_native;       // caller slot -> HFactor slot
    std::vector<HighsInt> to_caller;       // HFactor slot -> caller slot
    std::vector<HighsInt> slot_of_column;  // scratch for recovering the permutation
    bool mapped = false;
    // Every forward solve runs through aq and every transposed one through ep, so the vector an update
    // needs is the one the caller's own solve just left behind and no second solve is required to get it.
    HVector aq;
    HVector ep;
    double build_synthetic_tick = 0.0;
    /*
     * Every solve crossing this seam adds to the clock. HiGHS instead accumulates only its own iteration's
     * solves, so a caller doing extra solves per iteration wears this clock down faster than the same
     * workload would there, and kSyntheticTickReinversionMinUpdateCount counts against a different
     * denominator. A seam cannot know which solves are "the iteration's", so this is the honest reading of
     * the rule rather than HiGHS's, and a caller tuning against HiGHS's published behaviour should know it.
     */
    double total_synthetic_tick = 0.0;
    HighsInt update_count = 0;
    /* What the last update advised rebuilding for: 0 nothing, 1 HFactor's own hint, 2 the clock above. */
    int32_t refactorize_reason = 0;
    /* How much of the basis survived triangularization into the Markowitz kernel, as build left it. */
    HighsInt kernel_dim = 0;
    HighsInt kernel_num_el = 0;
    /*
     * The pivot range, filled on the first read after the factors change rather than on every read. Reading
     * it copies the whole factorization, so a caller that never asks pays nothing and one that asks twice
     * between updates pays once.
     */
    bool pivot_range_known = false;
    double smallest_pivot = 0.0;
    double largest_pivot = 0.0;
    /*
     * Fill, tracked rather than read back. HFactor hands its factors out only by copy, so asking it costs a
     * duplicate of every L and U array, and fill is what a caller reads once an iteration to pace its
     * rebuilds. build publishes its own count and an update adds the spike it packed, which is what the
     * Forrest-Tomlin path appends to U. Like the portable solver's count this only rises: an update also
     * deletes the pivotal column's old entries, and following that would cost the copy this avoids.
     */
    HighsInt fill = 0;
};

/*
 * Loads the caller's vector, whose storage is HVector's own: values dense over the dimension, positions of
 * the nonzeros in the first [count] entries of [index]. Positions map through [map] where the caller's
 * index space and HFactor's differ.
 */
void load(HVector& v, int32_t count, const int32_t* index, const double* array, const HighsInt* map) {
    v.clear();
    // The update reads the packed form the solve records, which is only kept when the flag is set going in.
    v.packFlag = true;
    for (int32_t k = 0; k < count; k++) {
        const HighsInt from = index[k];
        const HighsInt to = map == nullptr ? from : map[from];
        v.array[to] = array[from];
        v.index[k] = to;
    }
    v.count = count;
}

/*
 * Writes the solve back over the caller's vector in place. The positions it came in at are zeroed first,
 * since the result generally stands at others, and the caller's storage carries no nonzero it has not
 * named.
 */
int32_t store(const HVector& v, int32_t in_count, int32_t* index, double* array, const HighsInt* map) {
    for (int32_t k = 0; k < in_count; k++) array[index[k]] = 0.0;
    for (HighsInt k = 0; k < v.count; k++) {
        const HighsInt from = v.index[k];
        const HighsInt to = map == nullptr ? from : map[from];
        index[k] = to;
        array[to] = v.array[from];
    }
    return v.count;
}

void scatterColumn(Handle* h, HVector& v, HighsInt column) {
    v.clear();
    v.packFlag = true;
    for (HighsInt k = h->start[column]; k < h->start[column + 1]; k++) {
        v.array[h->index[k]] = h->value[k];
        v.index[v.count] = h->index[k];
        v.count++;
    }
}

} // namespace

/*
 * The library is built with hidden visibility so a process that also loads a real HiGHS does not find two
 * definitions of its internals; only these entry points are exported.
 */
#define KOBLAS_HFACTOR_EXPORT __attribute__((visibility("default")))

extern "C" {

/* The constraint matrix in CSC, whose columns every later basis is drawn from. */
KOBLAS_HFACTOR_EXPORT Handle* koblas_hfactor_create_v2(int32_t num_row, int32_t num_col, const int32_t* start, const int32_t* index,
                              const double* value, double pivot_threshold, double pivot_tolerance, int32_t update_method) {
    if (num_row < 0 || num_col < num_row) return nullptr;
    Handle* h = new (std::nothrow) Handle();
    if (h == nullptr) return nullptr;
    h->num_row = num_row;
    h->num_col = num_col;
    h->start.assign(start, start + num_col + 1);
    h->index.assign(index, index + start[num_col]);
    h->value.assign(value, value + start[num_col]);
    h->basic_index.assign(num_row, 0);
    /* Identity until a build establishes the real permutation. */
    h->to_native.resize(num_row);
    h->to_caller.resize(num_row);
    for (HighsInt t = 0; t < num_row; t++) h->to_native[t] = h->to_caller[t] = t;
    h->slot_of_column.assign(num_col, 0);
    /* HFactor retains the pointer, so basic_index is sized once here and never resized. */
    h->factor.setup(num_col, num_row, h->start.data(), h->index.data(), h->value.data(), h->basic_index.data(),
                    pivot_threshold, pivot_tolerance, kHighsDebugLevelMin, nullptr, true, update_method);
    h->aq.setup(num_row);
    h->ep.setup(num_row);
    return h;
}

KOBLAS_HFACTOR_EXPORT void koblas_hfactor_free(Handle* h) { delete h; }

/* Factorizes the basis of basic_index. Returns 0, or the rank deficiency HFactor found. */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_build(Handle* h, const int32_t* basic_index) {
    for (HighsInt t = 0; t < h->num_row; t++) h->basic_index[t] = basic_index[t];
    const HighsInt rank_deficiency = h->factor.build();
    h->build_synthetic_tick = h->factor.build_synthetic_tick;
    h->total_synthetic_tick = 0.0;
    h->update_count = 0;
    h->refactorize_reason = 0;
    h->pivot_range_known = false;
    h->kernel_dim = h->factor.kernel_dim;
    h->kernel_num_el = h->factor.kernel_num_el;
    h->fill = h->factor.invert_num_el;
    /*
     * build reorders basic_index into its own pivot order, so a caller's slot and HFactor's stop agreeing
     * here. The two maps carry every later solve and update between the orderings, which keeps this
     * binding's slots the ones the caller named. A rank-deficient basis is reported instead, and HFactor
     * has substituted logicals into basic_index, so there is no permutation of the caller's basis to
     * recover and the maps stay as they were.
     */
    h->mapped = rank_deficiency == 0;
    if (!h->mapped) return rank_deficiency;
    for (HighsInt s = 0; s < h->num_row; s++) h->slot_of_column[h->basic_index[s]] = s;
    for (HighsInt t = 0; t < h->num_row; t++) {
        const HighsInt s = h->slot_of_column[basic_index[t]];
        h->to_native[t] = s;
        h->to_caller[s] = t;
    }
    return 0;
}

/*
 * Factorizes like koblas_hfactor_build, but keeps the repair rather than refusing it.
 *
 * HFactor completes a rank-deficient factorization with unit pivots for the rows it found no pivot for, so
 * the factors it leaves are invertible; what they invert is a basis the caller did not ask for. That basis
 * is written to [repaired], one entry per slot: a column of the caller's matrix, or num_col + row for a
 * slot HFactor filled with the unit column of that row, which is not a column of the matrix at all.
 *
 * There is no permutation of the caller's basis to recover once slots have been replaced, so the caller's
 * slots become HFactor's and [repaired] is what says where everything sits. Returns the rank deficiency, or
 * 0 when the basis factorized as given and [repaired] is the caller's own basis unchanged.
 */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_build_repairing(Handle* h, const int32_t* basic_index,
                                                             int32_t* repaired) {
    const int32_t deficiency = koblas_hfactor_build(h, basic_index);
    if (deficiency == 0) {
        for (HighsInt t = 0; t < h->num_row; t++) repaired[t] = basic_index[t];
        return 0;
    }
    for (HighsInt t = 0; t < h->num_row; t++) {
        h->to_native[t] = t;
        h->to_caller[t] = t;
        repaired[t] = h->basic_index[t];
    }
    h->mapped = true;
    return deficiency;
}

/* Solves B x = b in place over the caller's vector. Returns the solution's nonzero count. */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_ftran(Handle* h, int32_t count, int32_t* index, double* array,
                                                   double expected_density) {
    /* The right-hand side is indexed by row and the solution by basis slot, so only the result maps. */
    load(h->aq, count, index, array, nullptr);
    h->factor.ftranCall(h->aq, expected_density);
    h->total_synthetic_tick += h->aq.synthetic_tick;
    return store(h->aq, count, index, array, h->to_caller.data());
}

/* Solves Bᵀ x = b in place, the transposed counterpart of koblas_hfactor_ftran. */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_btran(Handle* h, int32_t count, int32_t* index, double* array,
                                                   double expected_density) {
    /* Mirror of the forward solve: the right-hand side is indexed by basis slot and the solution by row. */
    load(h->ep, count, index, array, h->to_native.data());
    h->factor.btranCall(h->ep, expected_density);
    h->total_synthetic_tick += h->ep.synthetic_tick;
    return store(h->ep, count, index, array, nullptr);
}

/*
 * One Forrest-Tomlin update: basis slot pivot_row takes column entering.
 *
 * reuse_spike and reuse_pivot_eta say whether aq and ep still hold the caller's own solves for this pivot,
 * which is the ordinary case in a dual simplex and saves recomputing them. Where they do not, the solve is
 * redone here, since an update reads the packed form a solve records rather than the spike's values alone.
 *
 * Returns -1 when the pivot cannot be inverted, 1 when the factors are worn enough to want rebuilding, and
 * 0 otherwise.
 */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_update(Handle* h, int32_t pivot_row, int32_t entering, int32_t reuse_spike,
                              int32_t reuse_pivot_eta) {
    if (!reuse_spike) {
        scatterColumn(h, h->aq, entering);
        h->factor.ftranCall(h->aq, 1.0);
        h->total_synthetic_tick += h->aq.synthetic_tick;
    }
    const HighsInt slot = h->to_native[pivot_row];
    const double pivot = h->aq.array[slot];
    if (pivot == 0.0 || !std::isfinite(pivot)) return -1;

    if (!reuse_pivot_eta) {
        h->ep.clear();
        h->ep.packFlag = true;
        h->ep.count = 1;
        h->ep.index[0] = slot;
        h->ep.array[slot] = 1.0;
        h->factor.btranCall(h->ep, 1.0);
        h->total_synthetic_tick += h->ep.synthetic_tick;
    }

    HighsInt row = slot;
    HighsInt hint = 0;
    h->factor.update(&h->aq, &h->ep, &row, &hint);
    h->basic_index[slot] = entering;
    h->update_count++;
    h->fill += h->aq.packCount;
    h->pivot_range_known = false;

    /*
     * The Forrest-Tomlin path leaves hint alone, so the advice comes from HiGHS's own synthetic clock rule:
     * rebuild once the updates have cost what the factorization did, and not before a floor of them.
     */
    if (hint != 0) {
        h->refactorize_reason = 1;
        return 1;
    }
    const bool worn = h->total_synthetic_tick >= h->build_synthetic_tick;
    const bool rebuild = worn && h->update_count >= kSyntheticTickReinversionMinUpdateCount;
    h->refactorize_reason = rebuild ? 2 : 0;
    return rebuild ? 1 : 0;
}

/* Which of the two rules the last update's advisory came from, so a caller can tell them apart. */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_refactorize_reason(const Handle* h) { return h->refactorize_reason; }

/*
 * The kernel build left behind: its dimension and its stored entries. Both are counted during the
 * factorization and read here, so this says how much of the basis triangularization peeled off before
 * Markowitz had to choose pivots, which a fill count alone does not.
 */
KOBLAS_HFACTOR_EXPORT void koblas_hfactor_kernel(const Handle* h, int32_t* dimension, int32_t* entries) {
    *dimension = h->kernel_dim;
    *entries = h->kernel_num_el;
}

KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_update_count(const Handle* h) { return h->update_count; }

/* The tracked fill, which costs nothing to read and so can be read every iteration. */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_fill(const Handle* h) { return h->fill; }

/*
 * The pivot magnitudes, apart from the fill because this is the expensive half: reaching the pivots means a
 * copy of the whole factorization. Held until the factors move, so asking twice between updates costs one
 * copy and never asking costs none.
 */
KOBLAS_HFACTOR_EXPORT void koblas_hfactor_pivot_range(Handle* h, double* smallest_pivot,
                                                      double* largest_pivot) {
    if (!h->pivot_range_known) {
        const InvertibleRepresentation invert = h->factor.getInvert();
        double smallest = 0.0;
        double largest = 0.0;
        bool first = true;
        for (const double pivot : invert.u_pivot_value) {
            const double magnitude = std::fabs(pivot);
            if (first || magnitude < smallest) smallest = magnitude;
            if (magnitude > largest) largest = magnitude;
            first = false;
        }
        h->smallest_pivot = smallest;
        h->largest_pivot = largest;
        h->pivot_range_known = true;
    }
    *smallest_pivot = h->smallest_pivot;
    *largest_pivot = h->largest_pivot;
}
}
