# Module koblas

Double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform. Dense matrices are column-major and
sparse matrices are validated CSC.

Container conveniences use the shared engine [koblas][com.eignex.koblas.koblas], initialized from the
`koblas.engine` JVM property, `KOBLAS_ENGINE` environment variable or automatic platform selection.
Applications may assign it during ordered startup,
before library operations or concurrent work. Callers own safe publication; configuration and reads are
unsynchronized. [KoblasEngine][com.eignex.koblas.KoblasEngine] names
exact engines, such as portable Kotlin through [KoblasEngine.scalar][com.eignex.koblas.KoblasEngine.scalar].
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

Exact built-in engines for comparisons and vendor metadata also live in this package.
