# Module koblas

Double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform. Dense matrices are column-major and
sparse matrices are validated CSC.

Container conveniences use the shared engine [koblas][com.eignex.koblas.koblas], initialized from the
`koblas.engine` JVM property, `KOBLAS_ENGINE` environment variable or automatic platform selection.
Applications may assign it during ordered startup,
before library operations or concurrent work. Callers own safe publication; configuration and reads are
unsynchronized. [KoblasEngine][com.eignex.koblas.KoblasEngine] names
exact engines, such as portable Kotlin through [KoblasEngine.scalar][com.eignex.koblas.KoblasEngine.scalar].
BLAS routines are typed members of [KoblasEngine][com.eignex.koblas.KoblasEngine]. Engines share validation, alias
handling and scheduling, and differ only in arithmetic: rounding, FMA, exceptional results and tied-index choices
follow the selected implementation.

Factorizations and solvers are outside this library. Concurrent calls need separate mutable destinations and
separate [Scratch][com.eignex.koblas.Scratch].

# Package com.eignex.koblas

Dense and sparse containers, views, the default [koblas] engine, the [KoblasEngine] contract and [Scratch]
arrays. [Matrix] and [Vector] are mutable contracts with fixed shapes that custom types can implement.
[DenseMatrix] represents whole buffers and views through offsets and leading dimensions. [DenseVector]
represents contiguous buffers and slices with positive or negative strides. Dense equality and hashing compare
logical entries, and serialization writes those entries and decodes independent compact storage.
[MatrixStorage] and [VectorStorage] provide polymorphic serialization of built-in dense and sparse containers.
Products accept at most one sparse operand, on either side, and produce dense storage.
No sparse operand is densified. Products of two sparse matrices are unsupported. Rank updates and sparse
column dots are [KoblasEngine.ger] and [KoblasEngine.dotColumn] members.

The matrix and vector facades extend [Matrix] and [Vector]. [Matrix.copy] and [Vector.copy] return
independent storage; [Matrix.copyFrom] and [Vector.copyFrom] copy into existing, equally shaped storage
and handle overlapping inputs. [Matrix.transpose] preserves sparse storage and materializes other
implementations into dense storage. [Matrix.sum] provides total and axis-wise sums. Sparse writes
preserve ascending indices, including explicitly written zeros, and structural changes may replace
their buffers. [Vector.scale] mutates any vector storage through the default engine. Scaled vector addition uses
[KoblasEngine.axpy].
[KoblasEngine] owns typed dense and sparse BLAS operations; vectors carry their own addressing.
Raw array windows are internal kernel operations.

[Matrix.forEachStored] visits `(row, column, value)` in column order, with rows ascending within each
column. Sparse matrices visit stored entries including zeros; other implementations visit every entry.
Dense views use local indices and exclude padding. [Vector.forEachStored] provides the corresponding
index and value traversal for vectors.

[Vector.norm] and [Matrix.norm] share [Norm]. Vectors support [Norm.One], [Norm.Two] (the default),
[Norm.Infinity] and [Norm.MaxAbs], with the last two equivalent. Matrices support [Norm.Frobenius]
(the default), [Norm.One] for the largest absolute column sum, [Norm.Infinity] for the largest absolute
row sum and [Norm.MaxAbs] for the largest absolute entry. Matrix [Norm.Two] and vector [Norm.Frobenius]
are rejected. Norms accept dense, sparse and custom storage; empty inputs have norm zero for supported
choices, and sparse operands are never densified.

Exact built-in engines for comparisons and vendor metadata also live in this package.
