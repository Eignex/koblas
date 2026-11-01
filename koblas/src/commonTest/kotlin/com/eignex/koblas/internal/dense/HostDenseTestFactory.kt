package com.eignex.koblas.internal.dense

import com.eignex.koblas.internal.host.DenseHostSelection
import com.eignex.koblas.internal.host.HostDenseCall
import com.eignex.koblas.vendor.Blas

/** Force small fixtures through host adaptation independently of measured performance profiles. */
internal fun PortableDenseBlas(
    portable: PortableDenseBlas,
    host: Blas,
    eligible: Boolean = true,
    borrowed: Boolean = true,
): PortableDenseBlas = portable.withHost(
    DenseHostSelection { operation, _, _, _, view ->
        host.takeIf {
            eligible && (borrowed || !view) && HostDenseCall.entryPointFor(
                operation,
            ) in host.directlyImplemented
        }
    },
)
