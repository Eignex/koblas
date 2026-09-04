package com.eignex.koblas.sparse.host

import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.cholmod.CholmodConfig
import com.eignex.koblas.sparse.host.cholmod.CholmodSparseBlas
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu

/**
 * The independently configured sparse LU backends this native host can load from its host libraries.
 *
 * The parameters are in the order the JVM's own [F64SparseBackends] takes them, so a positional call means
 * the same thing on either target. KLU and UMFPACK were the first two here in the opposite order, which a
 * positional call could not tell apart; two constructors cannot offer both orders, since a call naming its
 * arguments would match either.
 *
 * The two are still separate classes rather than one `expect` and its `actual`s, since the halves they carry
 * are separate classes per target too. HFactor has no native binding, so this one carries four halves where
 * the JVM carries five.
 */
public class F64SparseBackends(
    kluConfig: KluConfig = KluConfig(),
    umfpackConfig: UmfpackConfig = UmfpackConfig(),
    basicluConfig: BasicluConfig = BasicluConfig(),
    cholmodConfig: CholmodConfig = CholmodConfig(),
) {
    /** The KLU sparse LU half. */
    public val klu: KluSparseLu = KluSparseLu(kluConfig)

    /** The UMFPACK sparse LU half. */
    public val umfpack: UmfpackSparseLu = UmfpackSparseLu(umfpackConfig)

    /** The BASICLU sparse LU half, the only one offering basis updates. */
    public val basiclu: BasicluSparseLu = BasicluSparseLu(basicluConfig)

    /** The CHOLMOD sparse matrix half, the one library here that carries a sparse product. */
    public val cholmod: CholmodSparseBlas = CholmodSparseBlas(cholmodConfig)
}
