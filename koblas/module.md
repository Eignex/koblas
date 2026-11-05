# Module koblas

Double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform. Dense matrices are column-major and
sparse matrices are validated CSC.

Container conveniences use the immutable default engine [koblas][com.eignex.koblas.koblas],
selected once from the platform's standard profile. [BuiltinEngines][com.eignex.koblas.engine.BuiltinEngines] names
exact engines, such as portable Kotlin through [BuiltinEngines.scalar][com.eignex.koblas.engine.BuiltinEngines.scalar].
Each BLAS routine is one member of [KoblasEngine][com.eignex.koblas.KoblasEngine]. Engines share validation, alias
handling and scheduling, and differ only in arithmetic: rounding, FMA, exceptional results and tied-index choices
follow the selected implementation.

Factorizations and solvers are outside this library. Concurrent calls need separate mutable destinations and
separate [Scratch][com.eignex.koblas.Scratch].

# Package com.eignex.koblas

Owning dense and sparse containers, vector views, the default [koblas] engine, the [KoblasEngine]
contract and [Scratch] arrays. [Matrix] and [Vector] are read-only contracts custom types can
implement. Products accept at most one sparse operand, on either side, and produce dense storage.
No sparse operand is densified. Products of two sparse matrices are unsupported. Windowed rank updates and
sparse column dots are [KoblasEngine.ger] and [KoblasEngine.dotColumn] members.

The matrix facade provides total and axis-wise [Matrix.sum] for dense, sparse and custom storage.
On [DenseMatrixRegion], [gemvInto] and [gemmInto] default to replacing the destination, [ger] and [syr]
default to unit scaling, and [trsv] and [trsm] solve triangular systems in place. These conveniences
use the default engine. Vector arguments are [DenseVector] views with exact logical lengths; their
offsets and strides are passed to the engine. [KoblasEngine] remains the full BLAS surface.

# Package com.eignex.koblas.engine

Exact engines, standard profiles, host platform facts and vendor metadata.
