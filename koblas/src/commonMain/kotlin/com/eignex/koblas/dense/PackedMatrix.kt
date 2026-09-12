package com.eignex.koblas.dense

/** The grouped dimension of a packed product operand. */
public enum class PackedRole {
    /** Row groups, then depth steps, then row lanes. */
    Left,

    /** Column groups, then depth steps, then column lanes. */
    Right,
}

/** Ownership of the strongly retained heap array. No native pointer is retained. */
public enum class PackedOwnership {
    /** Allocated by the packer and private to the operand. */
    Owned,

    /** Supplied by the caller, who controls mutation and synchronization. */
    Borrowed,
}

/**
 * Version one grouped FP64 format. Grouping is independent of compute microtiles and triangular solve order.
 * [depthStride] separates depth steps, [groupStride] separates groups, and [group] lanes are reserved per step.
 * Every padding entry in the [storageSize] extent is positive zero when produced by [ScalarLayoutKernels].
 * Heap arrays guarantee natural Double alignment only; [alignmentBytes] cannot promise a stronger alignment.
 * A nonzero [requiredSvlBytes] restricts exact consumers to that streaming length; zero is length independent.
 */
@Suppress("LongParameterList")
public class PackedMatrixLayout(
    /** Left or right grouped product role. */
    public val role: PackedRole,
    /** Logical row count. */
    public val rows: Int,
    /** Logical column count. */
    public val columns: Int,
    /** Number of contiguous lanes in each depth step. */
    public val group: Int,
    /** Physical distance between depth steps. */
    public val depthStride: Int = group,
    /** Physical distance between groups. */
    public val groupStride: Int = checkedStorageSize(
        (if (role == PackedRole.Left) columns else rows).toLong() * depthStride,
    ),
    /** Stable physical format identifier. */
    public val layoutId: Int = GROUPED_FP64,
    /** Physical format version. */
    public val version: Int = 1,
    /** Guaranteed storage alignment in bytes. */
    public val alignmentBytes: Int = Double.SIZE_BYTES,
    /** Required streaming vector length, or zero when length independent. */
    public val requiredSvlBytes: Int = 0,
) {
    /** Number of entries in the complete physical extent, including padding. */
    public val storageSize: Int

    init {
        require(rows >= 0 && columns >= 0 && group > 0) { "invalid packed dimensions or group" }
        require(layoutId == GROUPED_FP64 && version == 1) { "unsupported packed layout ID or version" }
        require(alignmentBytes == Double.SIZE_BYTES) { "heap packed storage requires natural Double alignment" }
        require(requiredSvlBytes == 0 || (requiredSvlBytes in 16..256 && requiredSvlBytes % 16 == 0)) {
            "invalid required streaming vector length"
        }
        val depth = if (role == PackedRole.Left) columns else rows
        val extent = if (role == PackedRole.Left) rows else columns
        require(depthStride >= group && groupStride.toLong() >= depth.toLong() * depthStride) {
            "packed strides overlap"
        }
        val groups = (extent.toLong() + group - 1) / group
        storageSize = checkedStorageSize(groups * groupStride)
    }

    internal fun index(row: Int, column: Int): Int {
        val lane = if (role == PackedRole.Left) row else column
        val depth = if (role == PackedRole.Left) column else row
        return lane / group * groupStride + depth * depthStride + lane % group
    }

    /** Rejects an incompatible consumer before it can mutate output. Zero [svlBytes] means unknown. */
    public fun requireCompatible(layoutId: Int, version: Int, svlBytes: Int = 0) {
        require(this.layoutId == layoutId && this.version == version) { "incompatible packed layout" }
        require(requiredSvlBytes == 0 || requiredSvlBytes == svlBytes) { "incompatible streaming vector length" }
    }

    /** Factories and format identifiers. */
    public companion object {
        /** Shared grouped FP64 format; geometry and strides are explicit in each descriptor. */
        public const val GROUPED_FP64: Int = 0x30001
    }
}

/**
 * Retained logical matrix with its own physical layout. Its heap array remains alive across calls and threads.
 * [bakedScale] and [bakedTranspose] describe transformations already present in its stored logical values;
 * consumers do not apply them again. Product alpha scales the represented matrix, including its baked scale.
 * Reuse unscaled operands when alpha changes. Borrowed storage must keep its contents and padding valid and
 * must not be mutated concurrently with a call. This object borrows no workspace and requires no close hook.
 */
public class PackedMatrix private constructor(
    internal override val buffer: DoubleArray,
    /** Validated physical format and geometry. */
    public val layout: PackedMatrixLayout,
    /** First physical entry of the window. */
    public val offset: Int,
    /** Who controls the retained array. */
    public val ownership: PackedOwnership,
    /** Scaling already present in logical stored values. */
    public val bakedScale: Double,
    /** Whether stored values already represent a transpose. */
    public val bakedTranspose: Boolean,
) : MatrixOperand() {
    override val rows: Int get() = layout.rows
    override val columns: Int get() = layout.columns

    init {
        require(offset >= 0 && offset.toLong() + layout.storageSize <= buffer.size) { "packed window exceeds storage" }
    }

    override fun value(row: Int, column: Int): Double = buffer[offset + layout.index(row, column)]

    /** Reads a logical entry without exposing owned mutable storage. */
    public operator fun get(row: Int, column: Int): Double {
        require(row in 0 until rows && column in 0 until columns) { "packed index out of bounds" }
        return value(row, column)
    }

    /** Factories and format identifiers. */
    public companion object {
        /**
         * Wraps existing caller-owned storage without copying. The descriptor validates the entire physical
         * extent; the caller guarantees the declared transforms and positive-zero padding describe its contents.
         */
        public fun wrap(
            data: DoubleArray,
            layout: PackedMatrixLayout,
            offset: Int = 0,
            bakedScale: Double = 1.0,
            bakedTranspose: Boolean = false,
        ): PackedMatrix = PackedMatrix(data, layout, offset, PackedOwnership.Borrowed, bakedScale, bakedTranspose)

        internal fun owned(layout: PackedMatrixLayout, scale: Double, transposed: Boolean): PackedMatrix =
            PackedMatrix(DoubleArray(layout.storageSize), layout, 0, PackedOwnership.Owned, scale, transposed)
    }
}

internal fun checkedStorageSize(size: Long): Int {
    require(size in 0..Int.MAX_VALUE.toLong()) { "storage size exceeds array capacity" }
    return size.toInt()
}
