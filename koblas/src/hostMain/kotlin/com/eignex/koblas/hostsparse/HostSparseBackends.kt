package com.eignex.koblas.hostsparse

import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse LU backends this native host can load from its host libraries. */
public class HostSparseBackends(umfpackConfig: UmfpackConfig = UmfpackConfig()) {
    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)
}
