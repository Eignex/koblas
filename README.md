# koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Kotlin has no standard multiplatform linear algebra library. koblas is a small one: serializable
matrix and vector containers, a well defined subset of double precision BLAS and LAPACK over them,
sparse factorizations, and a swappable compute backend that a native BLAS or GPU implementation can
drop into later without changing callers.

## What is supported

The dense operation set is a deliberate subset of double precision BLAS and LAPACK. Everything in
this table is implemented in portable Kotlin and tested against reference results on every target.

| Standard routine | koblas |
|---|---|
| ddot, daxpy, dscal | dot, axpy, scale (sparse aware) |
| dnrm2, dasum, idamax | norm2, asum, iamax |
| dcopy, dswap | copy, swap |
| dgemv, full alpha and beta form | LinearAlgebra.gemv |
| dger, rank one update | addOuter |
| dtrsv, dtrsm triangular solves | trsv, trsm |
| dgemm, full form with transpose flags | LinearAlgebra.gemm |
| dsyrk, symmetric rank k update | LinearAlgebra.syrk |
| dgetrf, dgetrs, LU factor and solve | factor, solve, plus determinant |
| dpotrf, dpotrs, dpotri, Cholesky family | cholesky, solveSpd, invertSpd |
| Cholesky rank one update and downdate | choleskyUpdateInPlace, choleskyDowndateInPlace |

There is also a materializing transpose, a matVec convenience over the view types, and the usual
ergonomic entry points (DenseMatrix.lu(), matMul, and so on).

Deviations from the standard are small and documented on each function. The notable ones: syrk has
no uplo parameter and always produces the full symmetric matrix, trsm solves from the left only, LU
solve takes a single right hand side (use trsm for blocks), cholesky regularizes non positive
definite pivots unless asked to be strict, and norm2 skips the overflow rescale so vector components
must stay within roughly 1e150. Every alpha and beta form follows the BLAS convention that beta
equal to zero overwrites the output without reading it.

Sparse support: SparseMatrix in compressed sparse column form, SparseLu with Markowitz threshold
pivoting and FTRAN and BTRAN solves, and EtaBasis, the product form of the inverse used for rank one
basis updates between refactorizations. This is the kernel a sparse simplex builds on.

All containers serialize with kotlinx.serialization.

## What is out of scope

Single precision, complex numbers, banded and packed storage layouts, right side trsm, QR, SVD, and
eigendecompositions. Level 2 routines with no consumer here, such as symv and trmv, are also out.
Nothing is supported silently: when a workload needs a new routine it gets implemented, tested, and
added to the table above.

## Usage

    import com.eignex.koblas.*

    // General dense solve via LU.
    val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
    val x = a.lu().solve(doubleArrayOf(3.0, 5.0))

    // Symmetric positive definite solve via Cholesky.
    val l = a.cholesky()
    val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))

    // Sparse solve: CSC matrix, sparse LU, FTRAN.
    val s = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
    val xsp = SparseLu.factorize(s)!!.ftran(doubleArrayOf(3.0, 5.0))

## Backends

There are two performance seams. The level 1 kernels (dot, axpy, scale) dispatch at compile time:
the JVM uses the incubator Vector API when started with --add-modules=jdk.incubator.vector and
scalar loops otherwise, and all other targets are scalar. The heavier operations (gemv, gemm, syrk,
LU) sit behind the runtime LinearAlgebra interface. Today every platform uses the portable reference
implementation; a native BLAS or GPU backend can replace it without changing callers, and any
replacement must match the reference on the BlasConformanceTest suite. Storage is flat, contiguous,
row major DoubleArray, so a native backend receives raw buffers with no repacking.

## Coordinates

    implementation("com.eignex:koblas:<version>")

## Building

    ./gradlew check lintDocs

## Status

Early. The operation set grows on demand, and native BLAS and GPU backends are planned behind the
LinearAlgebra seam.
