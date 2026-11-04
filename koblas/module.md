# Module koblas

Double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform. Dense matrices are column-major and
sparse matrices are validated CSC.

Operators and container conveniences use the immutable default engine [koblas][com.eignex.koblas.koblas],
selected once from the platform's standard profile. [BuiltinEngines][com.eignex.koblas.engine.BuiltinEngines] names
exact engines, such as portable Kotlin through [BuiltinEngines.scalar][com.eignex.koblas.engine.BuiltinEngines.scalar].
Each BLAS routine is one member of [KoblasEngine][com.eignex.koblas.KoblasEngine]. Engines share validation, alias
handling and scheduling, and differ only in arithmetic: rounding, FMA, exceptional results and tied-index choices
follow the selected implementation.

Factorizations and solvers are outside this library. Concurrent calls need separate mutable destinations and
separate [Scratch][com.eignex.koblas.Scratch].

# Package com.eignex.koblas

Owning dense and sparse containers, vector views, operators, the default [koblas] engine, the [KoblasEngine]
contract, [MatrixStructure] and [Scratch] arrays. [Matrix] and [Vector] are read-only contracts custom types can
implement. Products accept dense or sparse operands on either side: two sparse operands produce CSC storage, other
pairings produce dense storage, and no sparse operand is densified.

# Package com.eignex.koblas.dense

Dense routines over matrices, views and vector windows with offsets, column spacing and signed strides. `gemmt` is
the triangle-selected product Netlib calls `GEMMTR`.

# Package com.eignex.koblas.engine

Exact engines, standard profiles, host platform facts and vendor metadata.

# Package com.eignex.koblas.sparse

Sparse routines over validated CSC storage: an absent position is never evaluated, and a stored zero is multiplied
wherever the reference BLAS routine would multiply it.

