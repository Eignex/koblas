package com.eignex.koblas.internal.diagnostics

import com.eignex.koblas.*
import com.eignex.koblas.sparse.*

/** Snapshot facts interpreted by the call's engine, without initializing the orientation cache. */
internal fun preparedRoute(
    engine: KoblasEngine,
    prepared: PreparedSparseMatrix,
    operation: SparseMatrixOperation,
    call: SparseCall,
): SparseMatrixRoute {
    val route = engine.routeOf(
        operation,
        SparseCall(
            prepared.snapshot,
            call.alpha, call.beta, call.destinationElements, call.depth, call.updateRun, call.rightHandSides,
            call.transposeSparse, call.transposeDense, call.lower, call.destinationRun,
        ),
    )
    val orients = operation == SparseMatrixOperation.GemmSparse || operation == SparseMatrixOperation.GemmSparseDense
    if (!orients || !call.transposeSparse || route.kind == RouteKind.NoWork) return route
    return SparseMatrixRoute(
        route.operation,
        RouteKind.Composed,
        route.scheduling,
        route.entryPoint,
        route.components.filter { it.endsWith("/scale") },
        route.executionGroup,
        route.executionTail,
        resolved = false,
    )
}
