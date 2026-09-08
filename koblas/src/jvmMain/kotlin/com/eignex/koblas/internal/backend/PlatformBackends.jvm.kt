package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.F64BundledBackend
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Cblas
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import java.util.ServiceLoader

/**
 * The JVM's backend discovery, run once when [com.eignex.koblas.discoverBackends] is called. koblas's own
 * halves first, then [ServiceLoader] providers; [Backend.priority] picks the winner on each half.
 */
internal actual fun registerPlatformBackends() {
    val requested = requestedBackends()
    val automatic = AutomaticHostConfiguration()
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (automatic.overrides(provider)) continue
        val offered = offerFor(provider, requested)
        if (offered.isEmpty) continue
        if (!probe(provider, offered.halves)) continue
        // Once, not per half, since registerBackend offers the object as every half it implements, less
        // whatever a pin on one half left out.
        BackendRegistry.registerAutomatic(provider, offered)
    }
    registerBuiltins(automatic, requested)
}

/** Deployment overrides are read only while automatic discovery chooses its candidates. */
private class AutomaticHostConfiguration {
    val openBlas = HostBlasConfig(
        libraryPath = libraryPath(ConfigurationKeys.CBLAS_PATH),
    )
    val hfactor = HfactorConfig(libraryPath(ConfigurationKeys.HFACTOR_PATH))

    /** What a deployment pointed at a library of its own, read once off [ConfigurationKeys.LIBRARY_PATHS]. */
    private val configuredPaths: Map<String, List<String?>> =
        ConfigurationKeys.LIBRARY_PATHS.mapValues { (_, keys) -> keys.map(::libraryPath) }

    /**
     * Whether a configured library supersedes [provider]. Only a bundled provider steps aside: a configured
     * one is what it would step aside for.
     */
    fun overrides(provider: Backend): Boolean = provider is F64BundledBackend &&
        configuredPaths[provider.canonicalName].orEmpty().any { it != null }
}

/**
 * koblas's own FFM bindings, offered once each. Presence is a `dlopen` plus a symbol lookup, not a probe
 * computation, because `Linker.downcallHandle` is stack-hungry enough to throw StackOverflowError here;
 * dense correctness is covered by [probe] for service providers instead.
 *
 * The backends are built once for this pass and handed over, rather than looked up per question, since
 * constructing them twice would open the library twice.
 */
private fun registerBuiltins(automatic: AutomaticHostConfiguration, requested: Map<BackendSlot, String?>) {
    registerIfOffered(F64Cblas(automatic.openBlas), requested)
    registerIfOffered(F64SparseBackends(hfactorConfig = automatic.hfactor).hfactor, requested)
}

/** Instantiate all registered providers, dropping any whose construction fails. */
private fun loadProviders(): List<Backend> {
    val providers = ArrayList<Backend>()
    loadProviders(Backend::class.java, providers)
    return providers.distinctBy { it::class.java.name }
}

/** Adds providers from one service type, dropping a malformed registration without failing discovery. */
private fun <T : Backend> loadProviders(type: Class<T>, providers: MutableList<Backend>) {
    val iterator = ServiceLoader.load(type).iterator()
    while (true) {
        @Suppress("TooGenericExceptionCaught") // provider loading can throw Error (ServiceConfigurationError)
        try {
            if (!iterator.hasNext()) break
            providers.add(iterator.next())
        } catch (_: Throwable) {
            continue // ServiceLoader advances past the bad entry before reporting it
        }
    }
}

/**
 * Makes the candidate load its native path and produce a correct result.
 *
 * The primitive is called rather than the convenience overload, since a backend built by delegation
 * inherits a forwarder for every routine it does not override and the convenience one would bypass it.
 *
 * A candidate whose dense half is not among [offered] is asked only whether it loaded, since the product it
 * would be asked to compute is the half a pin already turned away.
 */
internal fun probe(backend: Backend, offered: Set<BackendSlot> = BackendSlot.entries.toSet()): Boolean {
    if (backend !is Blas || BackendSlot.Blas !in offered) return backend.isAvailable
    val n = 1
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val a = DenseMatrix(n, n, DoubleArray(n * n) { 2.0 })
        val c = DenseMatrix(n, n)
        backend.gemm(1.0, a, transposeA = false, a, transposeB = false, beta = 0.0, c = c)
        val expected = 4.0 * n
        c.data.all { it == expected }
    } catch (_: Throwable) {
        false
    }
}
