package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse LU backends this native host can load from its host libraries. */
public class F64HostSparseBackends(umfpackConfig: UmfpackConfig = UmfpackConfig()) {
    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)
}
