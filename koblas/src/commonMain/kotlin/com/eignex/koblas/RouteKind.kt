package com.eignex.koblas

/** How a call reached its result. */
public enum class RouteKind {
    /** The vendor entry point ran over these operands. What it cost to reach it is named by [com.eignex.koblas.vendor.CallRoute.adapter]. */
    Direct,

    /** The whole call was handed to another implementation, which [com.eignex.koblas.vendor.CallRoute.adapter] names. */
    Delegated,

    /** The result was assembled from more than one call, or from a vendor call plus arithmetic here. */
    Composed,

    /** The operation's own contract says there is nothing to do, so nothing ran. */
    NoWork,

    /** Nothing ran and nothing can; [com.eignex.koblas.vendor.CallRoute.reason] says why. */
    Unsupported,
}

