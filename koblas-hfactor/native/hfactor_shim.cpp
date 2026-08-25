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
    double total_synthetic_tick = 0.0;
    HighsInt update_count = 0;
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
KOBLAS_HFACTOR_EXPORT Handle* koblas_hfactor_create(int32_t num_row, int32_t num_col, const int32_t* start, const int32_t* index,
                              const double* value) {
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
    h->factor.setup(num_col, num_row, h->start.data(), h->index.data(), h->value.data(),
                    h->basic_index.data());
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

    /*
     * The Forrest-Tomlin path leaves hint alone, so the advice comes from HiGHS's own synthetic clock rule:
     * rebuild once the updates have cost what the factorization did, and not before a floor of them.
     */
    if (hint != 0) return 1;
    const bool worn = h->total_synthetic_tick >= h->build_synthetic_tick;
    return (worn && h->update_count >= kSyntheticTickReinversionMinUpdateCount) ? 1 : 0;
}

KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_update_count(const Handle* h) { return h->update_count; }

/*
 * The fill and the pivot range in one call: HFactor hands out its factors only by copy, so the two are
 * read together and this is sampled rather than polled.
 */
KOBLAS_HFACTOR_EXPORT int32_t koblas_hfactor_stats(const Handle* h, double* smallest_pivot,
                                                   double* largest_pivot) {
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
    *smallest_pivot = smallest;
    *largest_pivot = largest;
    return static_cast<int32_t>(invert.l_index.size() + invert.u_index.size());
}
}
