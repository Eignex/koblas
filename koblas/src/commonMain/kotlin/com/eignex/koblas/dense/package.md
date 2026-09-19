# Package com.eignex.koblas.dense

Dense BLAS contracts and the implementations behind them.

The surface splits at the level, not at the backend. [DenseVectorKernels] is Level 1: arithmetic over one run
of a `DoubleArray`, taken as a buffer, an origin, a length and a step. It stays portable Kotlin because a
foreign call over a single vector costs more than the arithmetic it carries, and because it has to keep
working on a host with no library installed.

[DenseBlas] is Level 2 and 3. Built-in engines implement it in common Kotlin, with validation, zero-multiplier
no-read behavior and alias staging before mutation. That shared implementation owns traversal, windows and
dependency order and is the same on every platform; the arithmetic inside a window is [DensePanelKernels].

[DensePanelKernels] is the panel seam. A caller describes logical work — [PanelWork.MultiDot] reduces several
columns against one vector, [PanelWork.ColumnUpdate] accumulates several into one destination window,
[PanelWork.CoupledDotUpdate] does both in one pass for a symmetric traversal, and [PanelWork.RankUpdate]
accumulates one window into several columns — and asks
[DensePanelKernels.executionGroup] how many columns to hand over at a time. The answer is the local backend's,
resolved from its own vector width and measured rules, and any other positive answer is equally correct. A
panel group and a SIMD lane count are different numbers.

[DenseProductKernels] is the same seam one level up. A matrix product with enough arithmetic to hide a copy
is cut into cache blocks, each operand block is copied into the grouped layout [PackedLayout] describes, and
[DenseProductKernels.productBlock] accumulates a destination window from the two packed panels in a register
tile of the backend's own shape. A product without that much arithmetic runs where its operands lie, as
panel work down each destination column with nothing copied. [com.eignex.koblas.KoblasEngine.denseRouteOf]
reports which of the two a call takes, together with the packing it performs and the tile bodies its blocks
reach, rather than leaving any of it to an engine's name. A [PackedMatrix] is that copy kept: a caller
multiplying one operand repeatedly packs it once and hands the panel to later products, which
[com.eignex.koblas.KoblasEngine.packLeft] and [com.eignex.koblas.KoblasEngine.packRight] produce.

A tile's rows and columns, a panel's execution group, a cache block and a SIMD lane count are four
independent numbers. The remaining Level 3 routines are still direct scalar traversal and call neither a
panel nor a tile, which the route also reports.

Installed host bindings expose the same operation family separately. Their arithmetic and exceptional behavior
belong to the resolved library, and their route reports its binary and entry point. A requested engine name is
never used as proof that such a binding ran.

The platform default is exposed through [com.eignex.koblas.koblas], selected once and immutable. Tests and
benchmarks reach exact portable and JVM SIMD compositions through [com.eignex.koblas.BuiltinEngines], and
[com.eignex.koblas.KoblasEngine.explain] names the component a [DenseOperation] of a given length and spacing
actually reaches, so an attribution cannot claim a kernel the call did not run.

Operands are the public containers. [com.eignex.koblas.DenseMatrix] is contiguous column-major, and
[com.eignex.koblas.DenseVector] covers both dense spacings, adjacent entries and a fixed step, because a
vendor takes either as a pointer and an increment. Structure and transposition travel beside an operand as
flags rather than in its type: [MatrixStructure] says which triangle is stored and whether the diagonal is
implied, and a transpose is a boolean the call passes through.

`gemmt` is the triangular-result general product Netlib calls `GEMMTR`. It uses ordinary full column-major
operands and a selected full-storage destination triangle, not conventional compact BLAS packed storage.
Explicit vendor bindings report whether they called `cblas_dgemmt` or composed the operation.
