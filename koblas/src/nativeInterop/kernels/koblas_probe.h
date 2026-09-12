#ifndef KOBLAS_PROBE_H
#define KOBLAS_PROBE_H
#include <stdint.h>

#define KOBLAS_API __attribute__((visibility("default")))
#define KOBLAS_ABI_VERSION 1u
#define KOBLAS_HOST 1u
#define KOBLAS_KERNEL 2u
#define KOBLAS_THREAD 3u

#define KOBLAS_OK 0u
#define KOBLAS_INVALID_ARGUMENT 1u
#define KOBLAS_BAD_VERSION 2u
#define KOBLAS_BUFFER_TOO_SMALL 3u
#define KOBLAS_NOT_FOUND 4u
#define KOBLAS_UNSUPPORTED_CPU 5u
#define KOBLAS_UNSUPPORTED_OS 6u
#define KOBLAS_NEEDS_ENABLEMENT 7u
#define KOBLAS_INCOMPATIBLE_CONTEXT 8u
#define KOBLAS_NOT_BUILT 9u

#define KOBLAS_SCALAR 1u
#define KOBLAS_SSE2 2u
#define KOBLAS_AVX2 3u
#define KOBLAS_NEON 4u
#define KOBLAS_FEATURE_SSE2 (1u << 0)
#define KOBLAS_FEATURE_AVX (1u << 1)
#define KOBLAS_FEATURE_AVX2 (1u << 2)
#define KOBLAS_FEATURE_NEON (1u << 3)
#define KOBLAS_FEATURE_SVE (1u << 4)
#define KOBLAS_FEATURE_SME (1u << 5)
#define KOBLAS_FEATURE_SME2 (1u << 6)
#define KOBLAS_FEATURE_SME_FP64 (1u << 7)
#define KOBLAS_FEATURE_AVX512 (1u << 8)
#define KOBLAS_FEATURE_AVX10 (1u << 9)
#define KOBLAS_FEATURE_ACE (1u << 10)
#define KOBLAS_FEATURE_AMX (1u << 11)
#define KOBLAS_F64 1u
#define KOBLAS_F32 2u
#define KOBLAS_I32 3u
#define KOBLAS_STATE_SVE 1u
#define KOBLAS_STATE_ZA 2u
#define KOBLAS_STATE_TILE 4u

#define KOBLAS_EXECUTION_SCALAR 0u
#define KOBLAS_EXECUTION_SIMD 1u
#define KOBLAS_EXECUTION_STREAMING 2u
#define KOBLAS_WIDTH_NONE 0u
#define KOBLAS_WIDTH_FIXED 1u
#define KOBLAS_WIDTH_SCALABLE 2u
#define KOBLAS_PRIMITIVE_VECTOR 1u
#define KOBLAS_PRIMITIVE_PRODUCT_TILE 2u
#define KOBLAS_PRIMITIVE_SOLVE_TILE 3u
#define KOBLAS_PRIMITIVE_UPDATE_SOLVE_TILE 4u
#define KOBLAS_TAIL_LENGTH 1u
#define KOBLAS_TAIL_DEPTH 2u
#define KOBLAS_TAIL_ROWS_COLUMNS 4u
#define KOBLAS_SEMANTICS_ZERO_WORK_NO_READ 1u
#define KOBLAS_SEMANTICS_ZERO_ALPHA_NO_READ 2u
#define KOBLAS_SEMANTICS_UNIT_DIAGONAL_NO_READ 4u
#define KOBLAS_SEMANTICS_ROBUST_NORM 8u
#define KOBLAS_SEMANTICS_ORDERED_COEFFICIENTS 16u
#define KOBLAS_SEMANTICS_DELAYED_OUTPUT 32u
#define KOBLAS_ADDRESS_CONTIGUOUS 1u
#define KOBLAS_ADDRESS_COLUMN_STRIDE 2u
#define KOBLAS_ADDRESS_VECTOR_STRIDE 4u
#define KOBLAS_ADDRESS_INDEXED 8u
#define KOBLAS_ADDRESS_PACKED_FOUR 16u
#define KOBLAS_ADDRESS_COLUMN_MAJOR_OUTPUT 32u
#define KOBLAS_LAYOUT_PACKED_FOUR_V1 0x10001u
#define KOBLAS_LAYOUT_COLUMN_MAJOR_FOUR_V1 0x20001u

/* Tail masks describe logical dimensions accepted at entry, not physical vector predication. LENGTH
 * permits arbitrary vector/panel length; DEPTH permits arbitrary product depth; ROWS_COLUMNS permits
 * valid_rows/order in 0..4. Packed operands still obey their padded four-double layout. Address masks
 * describe each operation's actual ABI: column stride is a panel column separation, vector stride is
 * a signed element increment, INDEXED is an explicit index array. DELAYED_OUTPUT means all input
 * reductions complete before output stores (dot4 and product tiles). ORDERED_COEFFICIENTS preserves
 * panel coefficient order, including zero multipliers. Other math/alias rules are operation-specific.
 */

/* All ABI words are uint32_t, aligned to 4 bytes, in native byte order. No pointers or C enums.
 * Callers initialize both version/size prefixes. The complete v1 request is required. Larger requests
 * are accepted only with zero extension bytes; reserved fields must be zero. Results require the full
 * v1 prefix, leave caller extension bytes untouched, and report the v1 size. Errors write nothing when
 * the result prefix is invalid; otherwise only the result header/status is written. Unknown versions,
 * queries and IDs are errors, never fallback. index enumerates the immutable catalog; nonzero kernel_id
 * selects exactly that ID. No ordinal or function address is a persistent identity.
 */
typedef struct {
    uint32_t abi_version, byte_size, query, index, kernel_id, reserved[3];
} koblas_probe_request_v1;

typedef struct {
    uint32_t abi_version, byte_size, query, status;
    uint32_t catalog_revision, kernel_count, architecture, operating_system;
    uint32_t cpu_vendor, cpu_family, cpu_model, validity;
    uint32_t hardware_features, hardware_features_hi, os_features, os_features_hi;
    uint32_t built_features, built_features_hi, kernel_id, operation;
    uint32_t variant, reason, required_features, required_features_hi;
    uint32_t a_type, b_type, accumulation_type, output_type;
    uint32_t numerical_mode, execution_mode, width_mode, register_bits;
    uint32_t logical_batch, unroll, accumulators, primitive;
    uint32_t tile_rows, tile_columns, depth_multiple, tail_modes;
    uint32_t left_layout, right_layout, output_layout, alignment;
    uint32_t scratch_bytes, semantics, addressing, process_state;
    uint32_t thread_state, call_state, ready_process_state, ready_thread_state;
    uint32_t ordinary_vl_bytes, streaming_vl_bytes, context_validity, tuning_key;
    uint32_t reserved[8];
} koblas_probe_result_v1;

/* HOST facts are cached at library load; THREAD is a read-only current-thread OS query. It never sets
 * vector lengths or permissions. hardware_features on Arm means OS-advertised hardware, not raw ID
 * registers (validity bit 1); x86 uses CPUID (bit 0). context_validity bits 0/1 mean valid SVE/SME lengths.
 * Unknown CPU models use tuning_key=0. Layout 0 is direct storage, 0x10001 is depth-major groups of four
 * doubles, 0x20001 is column-major four-row tile output. All current operands/accumulators are F64.
 * Width is machine register bits; scalar width is zero. logical_batch is elements per explicit source
 * loop iteration (per column for panels), unroll is its explicit software repetition, and accumulators
 * counts independent scalar FP64 chains, not machine registers. Compiler loop transformations and scalar
 * tails are not additional catalog variants. These schedule facts do not change required operand geometry.
 * State masks
 * distinguish process permission, thread preparation, and per-call ABI state. No current leaf needs
 * preparation; no preparation service is exported. Feature words preserve unknown future bits.
 */
KOBLAS_API int32_t koblas_probe_v1(const koblas_probe_request_v1 *request, koblas_probe_result_v1 *result);
#endif
