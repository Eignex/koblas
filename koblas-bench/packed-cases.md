# Fixed packed workload contract

Workload 5, fixture 2, CSV schema 4 are a new baseline. Historical reports remain unchanged and cannot be
relabelled: earlier tile fixtures depended on physical width, and vendor and Kotlin workload versions differed.

`cases.txt` owns every packed setting. Options follow `density, mode, physical, side, uplo, transA, transB, diag`,
then the following canonical order. All fields in this table are required on every tile/layout/block case,
including zeros and values equal to current defaults. The two layout descriptors form one paired configuration;
a one-sided layout operation uses only its named side. Inapplicable schedule dimensions must explicitly be zero.

| Field | Supported meaning |
| --- | --- |
| `work` | Immutable mathematical recipe: `gemm-add-v1` is C = C + A*B; `right-solve-v1` is X*T = B; `update-solve-v1` is X*T = B - A*R. Layout recipes are operation name plus `-v1`. |
| `physical` | Exactly `4x4` or `8x4`. Execution validates the actual engine geometry before allocating fixtures. No geometry is inferred from a species or tuning profile. |
| `leftLayout`, `rightLayout` | Exactly `depth-rows-v1`, `depth-columns-v1`; immutable formulas below. |
| `leftGroup`, `rightGroup` | Exactly the declared physical rows/columns. |
| `leftStride`, `rightStride` | Doubles per depth step, exactly the corresponding group. No extra depth padding. |
| `padding`, `alignment` | `zero` means positive-zero edge lanes; `8` means natural double element alignment. No stronger address alignment is requested or promised for moving JVM arrays. |
| `block` | `m x n x k` without spaces: one whole logical block. For layout cases m/n are the source rows/columns and k is its depth (n for left, m for right). |
| `panel` | Exactly k: one full reduction panel, including any depth tail. |
| `diagonal`, `rhs` | Solve order n and RHS row count m for right solves; `0` for other operations. |
| `batch` | `1`: one old tile per kernel call. Block loops remain inside the timer. |
| `variant` | `current-tile-v1`: existing scalar, JVM Vector, or C tile component selected by the runner. Unsupported geometry is an unsupported row, never a substitute. |
| `timing` | `raw-tile`, `packing-only`, `layout-only`, `prepacked-compute`, or `pack-plus-compute`, restricted by operation below. |

Each configuration is parsed and validated by both runners, even for a vendor-unsupported operation. Other
layouts, alignment, strides, batches, panel splitting, and block overrides are rejected. Structured layout
sources must be square. Packed dimensions are bounded to 4096 and m*n*k to 16777216 to bound harness allocation.
These restrictions describe the current harness, not future production APIs or arbitrary tile support.

## Immutable layout and fixture formulas

Generate A(m,k), B(k,n), and C(m,n) in column-major logical order from operand streams 1, 2, and 4 before
packing. `Fixtures` and the C runner use the same SplitMix64 seed/stream and triangular recipe. A triangle
uses operand 20, order n, and the selected uplo; unit-diagonal solve flags ignore stored diagonal values.
Physical padding never advances a random stream. Matrix coordinates start at offset zero, are contiguous,
and have their logical row count as leading dimension; scaling and transpose are fixed by the recipe above.
These are immutable recipe semantics rather than changing engine defaults. Non-packed cases retain their
existing operand streams/scaling/flag semantics, now labelled as policy experiments in configuration metadata.

For groups r/c and reduction depth k, left storage has ceil(m/r)*r*k doubles and right storage has
ceil(n/c)*c*k doubles. A(i,p) maps to `(i/r)*r*k + p*r + i%r`; B(p,j) maps to
`(j/c)*c*k + p*c + j%c`, using integer division. All other packed entries are positive zero. Triangle T(i,j)
uses `i*c+j` in c*c doubles. A tile output uses `i+j*r` in r*c doubles.

`physical_work` records physical tile visits and the extents of left, right, output, reset template, tile
scratch, and triangle buffers. Each reported extent times eight is its byte size. `tiles` is the number of
old calls per operation, not the amount of reused scratch. Raw tiles allocate output plus its reset template;
blocks allocate a logical m*n output/reset template and one reusable r*c scratch tile. Logical A/B source
matrices and fixture-generation temporaries are excluded from these physical extents and all timing.
Layout rows report the operated panel extent; writeback additionally uses a logical m*n destination.
Raw GEMM performs r*c*k product terms including zero padding. Block GEMM performs
ceil(m/r)*ceil(n/c)*r*c*k terms. Solves update only their logical m*n entries and preserve output padding.

## Timing and comparison

| Case / timing | Included per invocation | Prepared outside timing |
| --- | --- | --- |
| tile / `raw-tile` | Reset physical output, one existing GEMM/solve/fused call, consume output element zero | Logical fixtures and packing; buffers |
| pack / `packing-only` | One existing general/structured pack operation, consume panel element zero | Source, destination allocation |
| write or clear / `layout-only` | One writeback or padding-clear operation, consume element zero | Valid prepacked input and destination allocation; padding is already zero on repeated clears |
| block / `prepacked-compute` | Reset logical C; loops over every old tile; zero scratch; stage C edges; call tile; copy valid entries back; consume C element zero | Logical fixtures, A/B packing, reusable scratch |
| block / `pack-plus-compute` | Both A/B packing calls plus the entire prepacked-compute boundary | Logical fixtures and all reusable buffers |

Both block timings perform one logical C += A*B with alpha=beta=1. Each call starts from the same reset
fixture; no allocation is timed. They are distinct comparisons because retained panels change the work.
The old loops, dispatch/interface calls, FFM or pinning, edge staging, and writeback stay inside timing.
This is a benchmark block driver, not a future production block implementation.

`logical_id` uses the versioned mathematical recipe, logical dimensions, fixture and semantic flags, independent
of operation spelling or implementation symbol. `configuration` retains the declared physical strategy.
`actual_kernel` records the executed component, including portable layout and portable JVM solve components.
JVM C products/fused solves use the existing internal FFM bindings through benchmark-only friend access,
bypassing their production depth crossover; this adapter is removed when PR 07's exact block interface replaces
it. Production dispatch is unchanged. Full BLAS cases still measure the current engine policy and report
`policy-v1`, not an exact low-level leaf or ISA. Current C compiler multiversioning does not expose an ISA ID.

`compare.sh --mode fixed` requires the same logical recipe, physical configuration/work and timing; policy
experiments are excluded. `--mode logical` permits different physical strategies for a complete block boundary
or full operation. Raw tiles and format-specific layout operations still require equal physical work. Both modes
reject mismatched schema/workload/fixtures/timing/threads/warmups/target durations. Distinct physical strategies
are never pooled: logical mode emits separate pairs with both configurations and physical extents. Kernel
renaming does not affect identity. `--require-compatible` fails on unmatched strategies for shared logical
work, or reports with no compatible pair. Unsupported rows have no sample and cannot establish equivalence.

Vendors receive the same logical fixtures in column-major storage. Raw tile arithmetic has the distinct
`vendor-arithmetic` boundary and cannot join raw Koblas tile rows. Vendor DGEMM can compare with logical block
`prepacked-compute` (including its logical reset); vendor pack-plus-compute is unsupported because its private
packing is not the declared format. Vendor physical metadata describes its actual column-major buffers, while
the original case column retains the requested configuration. No vendor packing or raw tile parity is claimed.

CSV timing provenance also records harness, warmup target per iteration, fork index/count, and elapsed basis.
JMH warms each iteration for target_ns and reports average time; elapsed_ns is reconstructed as score times
measured operations. Native/vendor warmup batches target max(1 ms, target_ns/4), then calibrate their measured
batch size. Their elapsed_ns is a monotonic batch measurement. Runtime-specific warmup protocols are retained
as provenance; the comparator requires equal requested warmup counts and measurement targets, not identical
harness protocols. Stable before/after claims still require repeated warmed measurements on the same host.
