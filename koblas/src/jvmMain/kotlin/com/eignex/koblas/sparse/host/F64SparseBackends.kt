package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.cholmod.CholmodConfig
import com.eignex.koblas.sparse.host.cholmod.CholmodSparseBlas
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/** The independently configured sparse backends this JVM can load from its host libraries. */
public class F64SparseBackends(
    kluConfig: KluConfig = KluConfig(),
    umfpackConfig: UmfpackConfig = UmfpackConfig(),
    basicluConfig: BasicluConfig = BasicluConfig(),
    hfactorConfig: HfactorConfig = HfactorConfig(),
    cholmodConfig: CholmodConfig = CholmodConfig(),
) {
    /** The KLU sparse LU half. */
    public val klu: KluSparseLu = KluSparseLu(kluConfig)

    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)

    /** The BASICLU sparse LU half, which offers basis updates. */
    public val basiclu: BasicluSparseLu = BasicluSparseLu(basicluConfig)

    /** The HFactor sparse LU half, which offers basis updates and hypersparse solves. */
    public val hfactor: HfactorSparseLu = HfactorSparseLu(hfactorConfig)

    /** The CHOLMOD sparse matrix half, the one library here that carries a sparse product. */
    public val cholmod: CholmodSparseBlas = CholmodSparseBlas(cholmodConfig)
}
