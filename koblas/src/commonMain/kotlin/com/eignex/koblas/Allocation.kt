package com.eignex.koblas

/** The primitive type of one reusable managed scratch buffer. */
public enum class ScratchKind {
    /** A [DoubleArray] buffer. */
    F64,

    /** An [IntArray] buffer. */
    I32,
}

/** A reusable managed scratch requirement for one operation. */
public data class ScratchRequirement(
    /** Primitive buffer type. */
    public val kind: ScratchKind,
    /** Entries in each buffer. */
    public val size: Int,
    /** Simultaneously idle buffers required. */
    public val count: Int = 1,
) {
    init {
        require(size >= 0) { "scratch size must not be negative, got $size" }
        require(count >= 0) { "scratch count must not be negative, got $count" }
    }
}

/** The strongest per-call allocation statement an operation can make. */
public enum class AllocationGuarantee {
    /** The operation makes no allocation promise. */
    UNRESTRICTED,

    /** Managed allocations, if any, are bounded independently of the problem size. */
    NO_SIZE_DEPENDENT_MANAGED,

    /** The operation performs no managed allocation; its native provider may still allocate. */
    NO_MANAGED,

    /** Neither koblas nor its provider performs managed or native allocation during the operation. */
    NO_MANAGED_OR_NATIVE,
}

/** A caller's allocation requirement for one operation. */
public enum class AllocationPolicy {
    /** Execute without rejecting an allocation behavior. */
    ALLOW,

    /** Require [AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED] or stronger. */
    REQUIRE_NO_SIZE_DEPENDENT_MANAGED,

    /** Require [AllocationGuarantee.NO_MANAGED] or stronger. */
    REQUIRE_NO_MANAGED,

    /** Require [AllocationGuarantee.NO_MANAGED_OR_NATIVE]. */
    REQUIRE_NO_MANAGED_OR_NATIVE,
}

/**
 * A declared hot-path allocation contract. [scratch] must already be idle in the supplied [Workspace]
 * before the operation can provide [guarantee]; reserving it is intentionally a caller action.
 */
public data class AllocationCapability(
    public val guarantee: AllocationGuarantee,
    public val scratch: List<ScratchRequirement> = emptyList(),
) {
    init {
        // Scanned rather than hashed. The list is a handful of entries at most, and a set would cost a
        // HashSet plus a Pair and a boxed Int per requirement on a type that operations build to promise
        // they will not allocate.
        for (i in scratch.indices) {
            for (j in 0 until i) {
                require(scratch[j].kind != scratch[i].kind || scratch[j].size != scratch[i].size) {
                    "duplicate ${scratch[i].kind} scratch requirement of size ${scratch[i].size}"
                }
            }
        }
    }

    /** Whether this capability can honor [policy] with buffers currently idle in [workspace]. */
    public fun supports(policy: AllocationPolicy, workspace: Workspace? = null): Boolean {
        val required = policy.guarantee ?: return true
        if (guarantee.ordinal < required.ordinal) return false
        return scratch.all { workspace != null && workspace.available(it) >= it.count }
    }
}

/** A strict [AllocationPolicy] that an operation cannot honor. */
public class AllocationPolicyRejectedException(
    /** The caller requirement that was rejected. */
    public val policy: AllocationPolicy,
    /** The operation contract that could not satisfy [policy]. */
    public val capability: AllocationCapability,
) : IllegalStateException(
    "allocation policy $policy rejected: operation provides ${capability.guarantee} " +
        "with scratch ${capability.scratch}",
)

internal val AllocationPolicy.guarantee: AllocationGuarantee?
    get() = when (this) {
        AllocationPolicy.ALLOW -> null
        AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED -> AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED
        AllocationPolicy.REQUIRE_NO_MANAGED -> AllocationGuarantee.NO_MANAGED
        AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE -> AllocationGuarantee.NO_MANAGED_OR_NATIVE
    }

internal val unrestrictedAllocation: AllocationCapability = AllocationCapability(AllocationGuarantee.UNRESTRICTED)
internal val noSizeDependentManagedAllocation: AllocationCapability =
    AllocationCapability(AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED)
internal val noManagedAllocation: AllocationCapability = AllocationCapability(AllocationGuarantee.NO_MANAGED)
internal val noManagedOrNativeAllocation: AllocationCapability =
    AllocationCapability(AllocationGuarantee.NO_MANAGED_OR_NATIVE)
