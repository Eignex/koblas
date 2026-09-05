package com.eignex.koblas.sparse.host

import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.sparse.FactorsNotExposed

/**
 * Raises for a factor the binding keeps in a form it cannot hand back.
 *
 * Through the lifecycle rather than thrown directly, so a closed factorization says it is closed instead of
 * answering for a factor it no longer holds. BASICLU and HFactor keep a basis representation for updating
 * rather than an `L` and a `U`, which is what they are for.
 */
internal fun NativeOwnership.factorNotExposed(factor: String): Nothing = anchoring { throw FactorsNotExposed(factor) }
