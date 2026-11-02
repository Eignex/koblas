@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

/**
 * Apply a plane rotation (BLAS `drot`). Each pair `(x_i, y_i)` becomes `(c*x_i + s*y_i, c*y_i - s*x_i)`,
 * so both [x] and [y] are overwritten in place. Negative strides are supported. Overlapping inputs are
 * snapshotted before writing; at a shared physical entry the final write is from [y].
 *
 * The rotation arrives as its cosine and sine rather than through a generator. BLAS `drotg` produced those
 * two from a pair of scalars, along with the rotated length only it consumed, and is not part of Koblas:
 * `c = a / hypot(a, b)` and `s = b / hypot(a, b)` is the whole of it, and a caller who wants a different sign
 * convention than Netlib's, which is what LAPACK's own `dlartg` exists to provide, would have to write it out
 * anyway.
 */
public fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double): Unit = koblas.rot(x, y, c, s)
