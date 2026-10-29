# Migrating to the reduced BLAS API

This change removes unsupported backend and retained-packing APIs. Recompile callers against the new
artifact and apply the replacements below; binary compatibility and deprecated forwarding APIs are not
provided. Downstream repository changes are outside this PR stack.

| Earlier API or assumption | Migration |
|---|---|
| `PackedMatrix`, `PackedLayout`, `PackedRole`, `packLeft`, `packRight` and packed GEMM overloads | Retain the original `DenseMatrix` or `DenseMatrixRegion` operands and call ordinary `gemm`. Pass a reused `Workspace` for bounded scratch retention. The schedule selects transient packing when useful; the API no longer retains packed operands between calls. |
| Public `DensePanelKernels`, `DenseProductKernels`, `DenseTriangularKernels`, `PanelWork`, or engine component getters | Call the engine's dense BLAS operations. These arithmetic seams are internal. For comparisons, select `BuiltinEngines.scalar` or `BuiltinEngines.simd` and inspect `routeOf`; custom backend injection has no public replacement. Raw vector and indexed sparse kernels remain available. |
| `DenseMatrixOperation.GemmPacked`, `GemmPackedLeft`, `GemmPackedRight` | Query `DenseMatrixOperation.Gemm` with the ordinary call's shape and storage facts. |
| `KoblasEngineApi`, `@KoblasEngineApi`, `@OptIn(KoblasEngineApi::class)`, or Gradle compiler opt-ins for that marker | Delete the annotation, import and compiler setting. Exact engines, route queries and `vendor` metadata are directly callable. |
| Matrix route `reason` text | Inspect `kind`, `scheduling`, `entryPoint`, `components` and applicable staging facts. Matrix routes no longer expose prose reasons. Explicit vendor `CallRoute.reason` remains available. |
| `KoblasException`, `SingularMatrix`, `NotPositiveDefinite` | Remove catches around ordinary BLAS calls: these types were unused by BLAS implementations. Factorizations and solver diagnostics belong to the caller, which should define its own failure types. Argument validation remains. |
| Exact scalar bits, a particular tied maximum index, or uniform NaN/Inf behavior across engines | Test routine-specific BLAS behavior. Compare finite numerical results with suitable rounding tolerances and accept choices the routine leaves to the implementation. |

| Prepared operands selecting the engine that prepared them | Preparation owns data and cached structure only. Use `engine.gemm(alpha, a: Matrix, transposeA, b: Matrix, transposeB, beta, destination, workspace)` for explicit policy. `Matrix` and prepared convenience methods use the default `koblas`. `engine.routeOf` describes that same receiver engine without warming caches. |
| Dense matrix BLAS overloads requiring `DenseMatrix` | Pass `DenseMatrixRegion`, which includes existing contiguous matrices and borrowed views, for every dense routine and dense sparse-product operand/destination. Recompile Java callers and custom `DenseBlas`/`vendor.Blas` implementations against the addressed signatures. |
| Array-only dense matrix-vector and rank-update execution | Array conveniences remain thin forwards; use `DenseVector` for explicit offsets and signed strides. Disjoint logical regions of the same backing buffer do not require snapshots. |
| Triangular conveniences on `MatrixStorage` | The receiver is now readable `Matrix`, including dense views, prepared sparse snapshots and custom operands. Serialization markers and payloads are unchanged. |
| `com.eignex.koblas.vendor.RouteKind` | Import `com.eignex.koblas.RouteKind`; dense, sparse and explicit vendor routes share this vocabulary. |
| `vendor.exactArmRejection` | Benchmark admission is tooling policy in `koblas-bench`. Production bindings supply route, vendor identity, resolved symbol, direct exports and thread evidence. |
| `portableTest` / `jvmPortableTest` | Standard `test` compiles common and baseline-safe JVM tests once to Java 17. `desktopTest` compiles only `jvmDesktopTest` sources to Java 25 and associates with compiled baseline/main/MR outputs. Existing packaged CI task names remain available. |

For example, replace retained packing with an ordinary array-backed call:

```kotlin
val workspace = Workspace()
repeat(iterations) {
    koblas.gemm(1.0, a, false, b, false, 0.0, result, workspace)
}
```

Keep the original operands for repeated calls. A workspace retains bounded scratch rather than a reusable
packed copy of an operand. Separate concurrent calls need separate destinations and workspaces.

Application calls can keep using `koblas`, or call `KoblasEngine.portable()` for an immutable portable
engine. Selecting an exact engine does not replace the process default. No engine installation or provider
registration is required. The [backend guide](BACKENDS.md) describes optional native acceleration.

For a generic product on an exact engine, prepare data independently from its execution policy:

```kotlin
val snapshot = sparse.prepare()
val engine = KoblasEngine.portable()
engine.gemm(0.875, snapshot, true, other as Matrix, false, 0.0, destination, workspace)
val route = engine.routeOf(0.875, snapshot, true, other, false, 0.0, destination)
```

JVM tests load packaged production classes: Java 17 selects the root implementation and Java 25 selects
the versioned implementation. The Java 25 packaged entrypoint depends on the canonical `jvmTest` execution,
so invoking both entrypoints in one graph executes that suite once, including with `--rerun-tasks`.
