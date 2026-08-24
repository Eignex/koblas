package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse LU backends this JVM can load from its host libraries. */
public class F64SparseBackends(kluConfig: KluConfig = KluConfig(), umfpackConfig: UmfpackConfig = UmfpackConfig()) {
    /** The KLU sparse LU half. */
    public val klu: KluSparseLu = KluSparseLu(kluConfig)

    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)
}
