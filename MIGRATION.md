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
