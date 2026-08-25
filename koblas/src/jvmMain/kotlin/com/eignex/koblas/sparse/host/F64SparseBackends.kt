package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse LU backends this JVM can load from its host libraries. */
public class F64SparseBackends(
    kluConfig: KluConfig = KluConfig(),
    umfpackConfig: UmfpackConfig = UmfpackConfig(),
    basicluConfig: BasicluConfig = BasicluConfig(),
    hfactorConfig: HfactorConfig = HfactorConfig(),
) {
    /** The KLU sparse LU half. */
    public val klu: KluSparseLu = KluSparseLu(kluConfig)

    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)

    /** The BASICLU sparse LU half, which offers basis updates. */
    public val basiclu: BasicluSparseLu = BasicluSparseLu(basicluConfig)

    /** The HFactor sparse LU half, which offers basis updates and hypersparse solves. */
    public val hfactor: HfactorSparseLu = HfactorSparseLu(hfactorConfig)
}
