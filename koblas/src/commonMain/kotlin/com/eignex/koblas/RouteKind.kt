package com.eignex.koblas

import com.eignex.koblas.vendor.CallRoute

/** How a call reached its result. */
public enum class RouteKind {
    /** The vendor entry point ran over these operands. What it cost to reach it is named by [CallRoute.adapter]. */
    Direct,

    /** The whole call was handed to another implementation, which [CallRoute.adapter] names. */
    Delegated,

    /** The result was assembled from more than one call, or from a vendor call plus arithmetic here. */
    Composed,

    /** The operation's own contract says there is nothing to do, so nothing ran. */
    NoWork,

    /** Nothing ran and nothing can; [com.eignex.koblas.vendor.CallRoute.reason] says why. */
    Unsupported,
}
