package com.eignex.koblas.vendor

/**
 * The double-precision CBLAS operations Koblas binds, each with the symbol it resolves to.
 *
 * [Gemmt] is the one operation outside the standard. oneMKL, AOCL and OpenBLAS export it; Accelerate does not,
 * and ArmPL is not assumed to. A vendor that lacks it composes the result instead, which is why the route of a
 * `gemmt` call says which of the two happened rather than naming the operation alone.
 */
public enum class BlasOperation(
    /** The CBLAS symbol, unsuffixed and therefore LP64. */
    public val entryPoint: String,
    /** The BLAS level, used to group report rows. */
    public val level: Int,
) {
    /** `dot = xᵀ · y`. */
    Dot("cblas_ddot", 1),

    /** `y = alpha · x + y`. */
    Axpy("cblas_daxpy", 1),

    /** `x = alpha · x`. */
    Scal("cblas_dscal", 1),

    /** `y = x`. */
    Copy("cblas_dcopy", 1),

    /** `nrm2 = ‖x‖₂`, computed without avoidable overflow. */
    Nrm2("cblas_dnrm2", 1),

    /** `asum = Σ|xᵢ|`. */
    Asum("cblas_dasum", 1),

    /** The index of the first entry of largest magnitude. */
    Iamax("cblas_idamax", 1),

    /** Exchanges `x` and `y`. */
    Swap("cblas_dswap", 1),

    /** Applies a plane rotation to `x` and `y`. */
    Rot("cblas_drot", 1),

    /** `y = alpha · op(A) · x + beta · y`. */
    Gemv("cblas_dgemv", 2),

    /** `y = alpha · A · x + beta · y` for symmetric `A`. */
    Symv("cblas_dsymv", 2),

    /** `A = alpha · x · yᵀ + A`. */
    Ger("cblas_dger", 2),

    /** `A = alpha · x · xᵀ + A` in the selected triangle. */
    Syr("cblas_dsyr", 2),

    /** `A = alpha · (x · yᵀ + y · xᵀ) + A` in the selected triangle. */
    Syr2("cblas_dsyr2", 2),

    /** Solves `op(A) · x = b` for triangular `A`. */
    Trsv("cblas_dtrsv", 2),

    /** `x = op(A) · x` for triangular `A`. */
    Trmv("cblas_dtrmv", 2),

    /** `C = alpha · op(A) · op(B) + beta · C`. */
    Gemm("cblas_dgemm", 3),

    /** `C = alpha · A · B + beta · C` for symmetric `A`. */
    Symm("cblas_dsymm", 3),

    /** `C = alpha · A · Aᵀ + beta · C` in the selected triangle. */
    Syrk("cblas_dsyrk", 3),

    /** `C = alpha · (A · Bᵀ + B · Aᵀ) + beta · C` in the selected triangle. */
    Syr2k("cblas_dsyr2k", 3),

    /** `B = alpha · op(A) · B` for triangular `A`. */
    Trmm("cblas_dtrmm", 3),

    /** Solves `op(A) · X = alpha · B` for triangular `A`. */
    Trsm("cblas_dtrsm", 3),

    /** [Gemm] restricted to one triangle of `C`, exported by some vendors and composed on the rest. */
    Gemmt("cblas_dgemmt", 3),
    ;

    /** Whether every supported vendor is required to export [entryPoint]. */
    internal val required: Boolean get() = this != Gemmt
}
