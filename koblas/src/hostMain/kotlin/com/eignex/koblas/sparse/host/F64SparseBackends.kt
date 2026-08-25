package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse LU backends this native host can load from its host libraries. */
public class F64SparseBackends(
    umfpackConfig: UmfpackConfig = UmfpackConfig(),
    kluConfig: KluConfig = KluConfig(),
    basicluConfig: BasicluConfig = BasicluConfig(),
) {
    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)

    /** The KLU sparse LU half. */
    public val klu: KluSparseLu = KluSparseLu(kluConfig)

    /** The BASICLU sparse LU half, the only one offering basis updates. */
    public val basiclu: BasicluSparseLu = BasicluSparseLu(basicluConfig)
}
