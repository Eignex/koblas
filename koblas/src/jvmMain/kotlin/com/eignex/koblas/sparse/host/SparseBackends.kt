package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

/** The independently configured sparse backends this JVM can load from its host libraries. */
public class SparseBackends(hfactorConfig: HfactorConfig = HfactorConfig()) {
    /** The HFactor sparse LU half, which offers basis updates and hypersparse solves. */
    public val hfactor: HfactorSparseLu = HfactorSparseLu(hfactorConfig)
}
