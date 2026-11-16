@file:Suppress("LongParameterList", "TooManyFunctions") // the CBLAS double-precision Level 2 and 3 surface

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.kernels.VectorKernels
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.storage.dense.*

/**
 * A host library's Level 2 and 3 routines, checked and staged as every implementation is before its CBLAS call.
 *
 * A binding supplies the calls themselves, which receive operands already validated and free of overlap with
 * their destination. A routine the library does not export fails before any operand is touched.
 */
internal interface HostBlas : VectorKernels {
    fun cblasGemv(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
    )

    fun cblasSymv(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
    )

    fun cblasGer(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
    )

    fun cblasSyr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
    )

    fun cblasSyr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
    )

    /** `cblas_dtrsv` or `cblas_dtrmv`, as [operation] names, which share one signature. */
    fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
    )

    fun cblasGemm(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeB: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
    )

    fun cblasGemmt(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeB: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    )

    fun cblasSymm(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        rightSide: Boolean,
    )

    fun cblasSyrk(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    )

    fun cblasSyr2k(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeA: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    )

    /** `cblas_dtrsm` or `cblas_dtrmm`, as [operation] names, which share one signature. */
    fun cblasTriangularMatrix(
        operation: EngineImplementation,
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        rightSide: Boolean,
    )
}
