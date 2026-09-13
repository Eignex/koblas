#include "internal.h"
#include <stddef.h>
#include <string.h>
#if defined(__x86_64__)
#include <cpuid.h>
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <sys/prctl.h>
#elif defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

_Static_assert(sizeof(koblas_probe_request_v1) == 32, "request ABI size");
_Static_assert(sizeof(koblas_probe_result_v1) == 256, "result ABI size");
_Static_assert(_Alignof(koblas_probe_result_v1) == 4, "result ABI alignment");
_Static_assert(offsetof(koblas_probe_result_v1, kernel_id) == 72, "kernel ABI offset");
_Static_assert(offsetof(koblas_probe_result_v1, ordinary_vl_bytes) == 208, "thread ABI offset");

typedef struct { uint32_t operation, batch, unroll, accumulators, tile; } operation;
static const operation operations[] = {
#include "catalog.inc"
};
static const uint32_t variants[] = {
    KOBLAS_SCALAR,
#if defined(KOBLAS_BUILD_SSE2)
    KOBLAS_SSE2,
#endif
#if defined(KOBLAS_BUILD_AVX2)
    KOBLAS_AVX2,
#endif
#if defined(KOBLAS_BUILD_NEON)
    KOBLAS_NEON,
#endif
};
#define OPERATION_COUNT (sizeof(operations) / sizeof(operations[0]))
#define VARIANT_COUNT (sizeof(variants) / sizeof(variants[0]))
static koblas_probe_result_v1 host;

/* Irregular indexed work has one explicit non-vectorized implementation. */
static int scalar_only(uint32_t op) { return op >= KOBLAS_OP_SPARSE_DOT_DENSE && op <= KOBLAS_OP_SPARSE_GATHER_ZERO; }

static uint32_t required_features(uint32_t variant) {
    switch (variant) {
    case KOBLAS_SCALAR: return 0;
    case KOBLAS_SSE2: return KOBLAS_FEATURE_SSE2;
    case KOBLAS_AVX2: return KOBLAS_FEATURE_AVX | KOBLAS_FEATURE_AVX2;
    case KOBLAS_NEON: return KOBLAS_FEATURE_NEON;
    default: return UINT32_MAX;
    }
}

#if defined(__APPLE__) && defined(__aarch64__)
static int feature(const char *name) {
    int enabled = 0;
    size_t size = sizeof(enabled);
    return sysctlbyname(name, &enabled, &size, NULL, 0) == 0 && size == sizeof(enabled) && enabled != 0;
}
#endif

/* Runs during load, outside any critical FFM compute call. No initializer executes arithmetic from a
 * targeted object. This cached record is immutable after loading; per-thread facts never go into it. */
__attribute__((constructor)) static void initialize_host(void) {
    host.abi_version = KOBLAS_ABI_VERSION;
    host.byte_size = sizeof(host);
    host.catalog_revision = 1;
    host.kernel_count = (OPERATION_COUNT - 7) * VARIANT_COUNT + 7;
#if defined(__linux__)
    host.operating_system = 1;
#elif defined(__APPLE__)
    host.operating_system = 2;
#endif
    for (size_t i = 0; i < VARIANT_COUNT; i++) host.built_features |= required_features(variants[i]);
#if defined(__x86_64__)
    host.architecture = 1;
    host.validity = 1;
    unsigned a, b, c, d;
    unsigned maximum = __get_cpuid_max(0, NULL);
    __cpuid(0, a, b, c, d);
    if (b == 0x756e6547 && d == 0x49656e69 && c == 0x6c65746e) host.cpu_vendor = 1;
    if (b == 0x68747541 && d == 0x69746e65 && c == 0x444d4163) host.cpu_vendor = 2;
    if (maximum >= 1) {
        __cpuid(1, a, b, c, d);
        unsigned family = (a >> 8) & 15u, model = (a >> 4) & 15u;
        host.cpu_family = family == 15 ? family + ((a >> 20) & 255u) : family;
        host.cpu_model = (family == 6 || family == 15) ? model + (((a >> 16) & 15u) << 4) : model;
        if (d & (1u << 26)) host.hardware_features |= KOBLAS_FEATURE_SSE2;
        if (c & (1u << 28)) host.hardware_features |= KOBLAS_FEATURE_AVX;
        host.os_features = host.hardware_features & KOBLAS_FEATURE_SSE2;
        uint32_t xcr0 = 0;
        if ((c & ((1u << 26) | (1u << 27))) == ((1u << 26) | (1u << 27))) {
            uint32_t high;
            __asm__ volatile("xgetbv" : "=a"(xcr0), "=d"(high) : "c"(0));
        }
        if (maximum >= 7) {
            __cpuid_count(7, 0, a, b, c, d);
            if (b & (1u << 5)) host.hardware_features |= KOBLAS_FEATURE_AVX2;
            if (b & (1u << 16)) host.hardware_features |= KOBLAS_FEATURE_AVX512;
            if (d & (1u << 24)) host.hardware_features |= KOBLAS_FEATURE_AMX;
        }
        if ((xcr0 & 6u) == 6u) host.os_features |= host.hardware_features &
            (KOBLAS_FEATURE_AVX | KOBLAS_FEATURE_AVX2);
        if ((xcr0 & 0xe6u) == 0xe6u) host.os_features |= host.hardware_features & KOBLAS_FEATURE_AVX512;
        /* AMX hardware does not establish dynamic permission or task readiness. */
    }
#elif defined(__aarch64__)
    host.architecture = 2;
    host.validity = 2;
#if defined(__linux__)
    /* Linux UAPI bit positions also work with the older supported Kotlin/Native sysroot. */
    unsigned long hw = getauxval(16), hw2 = getauxval(26);
    if (hw & (1ul << 1)) host.os_features |= KOBLAS_FEATURE_NEON;
    if (hw & (1ul << 22)) host.os_features |= KOBLAS_FEATURE_SVE;
    if (hw2 & (1ul << 23)) host.os_features |= KOBLAS_FEATURE_SME;
    if (hw2 & (1ul << 25)) host.os_features |= KOBLAS_FEATURE_SME_FP64;
    if (hw2 & (1ul << 37)) host.os_features |= KOBLAS_FEATURE_SME2;
#elif defined(__APPLE__)
    if (feature("hw.optional.neon")) host.os_features |= KOBLAS_FEATURE_NEON;
    if (feature("hw.optional.arm.FEAT_SVE")) host.os_features |= KOBLAS_FEATURE_SVE;
    if (feature("hw.optional.arm.FEAT_SME")) host.os_features |= KOBLAS_FEATURE_SME;
    if (feature("hw.optional.arm.FEAT_SME2")) host.os_features |= KOBLAS_FEATURE_SME2;
    if (feature("hw.optional.arm.FEAT_SME_F64F64")) host.os_features |= KOBLAS_FEATURE_SME_FP64;
#endif
    host.hardware_features = host.os_features;
#endif
}

uint32_t koblas_kernel_reason(uint32_t id, uint32_t op) {
    if (op == 0 || op > OPERATION_COUNT || id / 16u != op) return KOBLAS_NOT_FOUND;
    uint32_t variant = id % 16u;
    if (scalar_only(op) && variant != KOBLAS_SCALAR) return KOBLAS_NOT_BUILT;
    size_t i = 0;
    for (; i < VARIANT_COUNT && variants[i] != variant; i++) {}
    if (i == VARIANT_COUNT) return KOBLAS_NOT_BUILT;
    uint32_t required = required_features(variant);
    if ((host.hardware_features & required) != required) return KOBLAS_UNSUPPORTED_CPU;
    if ((host.os_features & required) != required) return KOBLAS_UNSUPPORTED_OS;
    return KOBLAS_OK;
}

int32_t koblas_probe_v1(const koblas_probe_request_v1 *request, koblas_probe_result_v1 *result) {
    if (!result || !request) return KOBLAS_INVALID_ARGUMENT;
    if (result->abi_version != KOBLAS_ABI_VERSION) return KOBLAS_BAD_VERSION;
    if (result->byte_size < sizeof(*result)) return KOBLAS_BUFFER_TOO_SMALL;
    uint32_t status = KOBLAS_OK;
    if (request->abi_version != KOBLAS_ABI_VERSION) status = KOBLAS_BAD_VERSION;
    else if (request->byte_size < sizeof(*request)) status = KOBLAS_BUFFER_TOO_SMALL;
    else {
        const unsigned char *bytes = (const unsigned char *)request;
        for (size_t i = offsetof(koblas_probe_request_v1, reserved); i < request->byte_size; i++) {
            if (bytes[i]) { status = KOBLAS_INVALID_ARGUMENT; break; }
        }
    }
    if (status != KOBLAS_OK) {
        result->abi_version = KOBLAS_ABI_VERSION; result->byte_size = sizeof(*result);
        result->query = 0; result->status = status;
        return (int32_t)status;
    }
    koblas_probe_result_v1 value = host;
    value.query = request->query;
    if (request->query == KOBLAS_HOST || request->query == KOBLAS_THREAD) {
        if (request->index || request->kernel_id) status = KOBLAS_INVALID_ARGUMENT;
        if (request->query == KOBLAS_THREAD) {
#if defined(__linux__) && defined(__aarch64__)
            if (host.os_features & KOBLAS_FEATURE_SVE) {
                int vl = prctl(51, 0, 0, 0, 0);
                if (vl >= 0) { value.ordinary_vl_bytes = (uint32_t)vl & 0xffffu; value.context_validity |= 1; }
            }
            if (host.os_features & KOBLAS_FEATURE_SME) {
                int vl = prctl(64, 0, 0, 0, 0);
                if (vl >= 0) { value.streaming_vl_bytes = (uint32_t)vl & 0xffffu; value.context_validity |= 2; }
            }
#endif
        }
    } else if (request->query == KOBLAS_KERNEL) {
        uint32_t id = request->kernel_id;
        if (id && request->index) status = KOBLAS_INVALID_ARGUMENT;
        if (!id) {
            if (request->index >= host.kernel_count) status = KOBLAS_NOT_FOUND;
            else {
                uint32_t remaining = request->index;
                for (size_t i = 0; i < OPERATION_COUNT; i++) {
                    uint32_t count = scalar_only(operations[i].operation) ? 1 : VARIANT_COUNT;
                    if (remaining < count) { id = operations[i].operation * 16u + variants[remaining]; break; }
                    remaining -= count;
                }
            }
        }
        uint32_t op = id / 16u, variant = id % 16u;
        if (status == KOBLAS_OK) {
            if (op == 0 || op > OPERATION_COUNT) status = KOBLAS_NOT_FOUND;
            else {
                const operation *entry = &operations[op - 1];
                value.kernel_id = id; value.operation = op; value.variant = variant;
                value.reason = koblas_kernel_reason(id, op);
                value.required_features = required_features(variant);
                value.a_type = value.b_type = value.accumulation_type = value.output_type = KOBLAS_F64;
                value.execution_mode = variant == KOBLAS_SCALAR ? KOBLAS_EXECUTION_SCALAR : KOBLAS_EXECUTION_SIMD;
                value.width_mode = variant == KOBLAS_SCALAR ? KOBLAS_WIDTH_NONE : KOBLAS_WIDTH_FIXED;
                value.register_bits = variant == KOBLAS_SCALAR ? 0 : variant == KOBLAS_AVX2 ? 256 : 128;
                value.logical_batch = entry->batch;
                value.unroll = entry->unroll;
                value.accumulators = entry->accumulators;
                if (variant == KOBLAS_SCALAR && !entry->tile && !scalar_only(op) && op != KOBLAS_OP_DENSE_NRM2) {
                    value.logical_batch = value.unroll = value.accumulators = 1;
                    if (op == KOBLAS_OP_DENSE_DOT4) value.accumulators = 4;
                }
#if defined(__aarch64__)
                if (op == KOBLAS_OP_DENSE_SCALE) value.logical_batch = value.unroll = value.accumulators = 1;
#endif
                value.primitive = entry->tile ? KOBLAS_PRIMITIVE_PRODUCT_TILE : KOBLAS_PRIMITIVE_VECTOR;
                if (op == KOBLAS_OP_DENSE_TRSM_TILE) value.primitive = KOBLAS_PRIMITIVE_SOLVE_TILE;
                if (op == KOBLAS_OP_DENSE_GEMM_TRSM_TILE) value.primitive = KOBLAS_PRIMITIVE_UPDATE_SOLVE_TILE;
                value.tile_rows = value.tile_columns = entry->tile ? 4 : 0;
                value.depth_multiple = 1;
                value.tail_modes = entry->tile ? KOBLAS_TAIL_DEPTH : KOBLAS_TAIL_LENGTH;
                if (op == KOBLAS_OP_DENSE_TRSM_TILE) value.tail_modes = KOBLAS_TAIL_ROWS_COLUMNS;
                if (op == KOBLAS_OP_DENSE_GEMM_TRSM_TILE) value.tail_modes |= KOBLAS_TAIL_ROWS_COLUMNS;
                value.left_layout = value.right_layout = entry->tile ? KOBLAS_LAYOUT_PACKED_FOUR_V1 : 0;
                if (op == KOBLAS_OP_DENSE_TRSM_TILE) value.left_layout = 0;
                value.output_layout = entry->tile ? KOBLAS_LAYOUT_COLUMN_MAJOR_FOUR_V1 : 0;
                value.alignment = 8;
                value.semantics = KOBLAS_SEMANTICS_ZERO_WORK_NO_READ;
                if (op == KOBLAS_OP_DENSE_AXPY || op == KOBLAS_OP_SPARSE_AXPY)
                    value.semantics |= KOBLAS_SEMANTICS_ZERO_ALPHA_NO_READ;
                if (op == KOBLAS_OP_DENSE_NRM2 || op == KOBLAS_OP_SPARSE_NRM2)
                    value.semantics |= KOBLAS_SEMANTICS_ROBUST_NORM;
                if (op == KOBLAS_OP_DENSE_AXPY4) value.semantics |= KOBLAS_SEMANTICS_ORDERED_COEFFICIENTS;
                if (op == KOBLAS_OP_DENSE_DOT4 || op == KOBLAS_OP_DENSE_GEMM_TILE)
                    value.semantics |= KOBLAS_SEMANTICS_DELAYED_OUTPUT;
                if (op == KOBLAS_OP_DENSE_TRSM_TILE || op == KOBLAS_OP_DENSE_GEMM_TRSM_TILE)
                    value.semantics |= KOBLAS_SEMANTICS_UNIT_DIAGONAL_NO_READ;
                value.addressing = scalar_only(op) ? KOBLAS_ADDRESS_INDEXED : KOBLAS_ADDRESS_CONTIGUOUS;
                if (op == KOBLAS_OP_DENSE_DOT4 || op == KOBLAS_OP_DENSE_AXPY4)
                    value.addressing |= KOBLAS_ADDRESS_COLUMN_STRIDE;
                if (op == KOBLAS_OP_DENSE_ROTM) value.addressing = KOBLAS_ADDRESS_VECTOR_STRIDE;
                if (entry->tile) value.addressing = KOBLAS_ADDRESS_PACKED_FOUR | KOBLAS_ADDRESS_COLUMN_MAJOR_OUTPUT;

            }
        }
    } else status = KOBLAS_INVALID_ARGUMENT;
    value.status = status;
    if (status == KOBLAS_OK) memcpy(result, &value, sizeof(value));
    else {
        result->abi_version = KOBLAS_ABI_VERSION; result->byte_size = sizeof(*result);
        result->query = request->query; result->status = status;
    }
    return (int32_t)status;
}
