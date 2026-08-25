package com.eignex.koblas.internal.backend

import com.eignex.koblas.Backend
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Backends
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import java.util.ServiceLoader

/**
 * The JVM's backend discovery, run once when [com.eignex.koblas.discoverBackends] is called. koblas's own
 * halves first, then [ServiceLoader] providers; [Backend.priority] picks the winner on each half.
 */
internal actual fun registerPlatformBackends() {
    val denseRequested = pinnedBackend(
        System.getProperty(ConfigurationKeys.DENSE_BACKEND_PROPERTY),
        System.getenv(ConfigurationKeys.DENSE_BACKEND_ENVIRONMENT),
    )
    val sparseRequested = pinnedBackend(
        System.getProperty(ConfigurationKeys.SPARSE_BACKEND_PROPERTY),
        System.getenv(ConfigurationKeys.SPARSE_BACKEND_ENVIRONMENT),
    )
    val automatic = AutomaticHostConfiguration()
    for (provider in loadProviders().sortedByDescending { it.priority }) {
        if (automatic.overrides(provider)) continue
        if (provider is F64Blas && denseRequested != null &&
            !matchesRequested(provider.name, denseRequested)
        ) {
            continue
        }
        if (provider is F64SparseKernels || provider is F64SparseBlas || provider is F64SparseLu) {
            if (sparseRequested != null &&
                !matchesRequested(provider.name, sparseRequested)
            ) {
                continue
            }
        }
        if (!probe(provider)) continue
        // Once, not per half, since registerBackend offers the object as every half it implements.
        BackendRegistry.registerAutomatic(provider)
    }
    registerBuiltins(automatic, denseRequested, sparseRequested)
}

/** Deployment overrides are read only while automatic discovery chooses its candidates. */
private class AutomaticHostConfiguration {
    val openBlas = HostBlasConfig(
        libraryPath = libraryPath(ConfigurationKeys.CBLAS_PATH),
        lapackeLibraryPath = libraryPath(ConfigurationKeys.LAPACKE_PATH),
    )
    val klu = KluConfig(libraryPath(ConfigurationKeys.KLU_PATH))
    val umfpack = UmfpackConfig(libraryPath = libraryPath(ConfigurationKeys.UMFPACK_PATH))

    fun overrides(provider: Backend): Boolean = when (provider.name.removeSuffix("-bundled")) {
        BackendNames.OPENBLAS -> provider.name.endsWith("-bundled") &&
            (openBlas.libraryPath != null || openBlas.lapackeLibraryPath != null)

        BackendNames.KLU -> provider.name.endsWith("-bundled") && klu.libraryPath != null

        BackendNames.UMFPACK -> provider.name.endsWith("-bundled") && umfpack.libraryPath != null

        else -> false
    }
}

private fun libraryPath(keys: LibraryPathKeys): String? = System.getProperty(keys.property)?.takeIf(::isAbsolutePath)
    ?: System.getenv(keys.environment)?.takeIf(::isAbsolutePath)

private fun isAbsolutePath(path: String): Boolean = java.nio.file.Path.of(path).isAbsolute

/** Bundled providers add a diagnostic suffix while retaining the canonical name callers configure. */
private fun matchesRequested(name: String, requested: String): Boolean =
    name == requested || name.removeSuffix("-bundled") == requested

/**
 * koblas's own FFM bindings, offered once each. Presence is a `dlopen` plus a symbol lookup, not a probe
 * computation, because `Linker.downcallHandle` is stack-hungry enough to throw StackOverflowError here;
 * dense correctness is covered by [probe] for service providers instead.
 *
 * The backends are built once for this pass and handed over, rather than looked up per question, since
 * constructing them twice would open the library twice.
 */
private fun registerBuiltins(
    automatic: AutomaticHostConfiguration,
    denseRequested: String?,
    sparseRequested: String?,
) {
    val dense = F64Backends(automatic.openBlas)
    registerIfOffered(dense.blas, denseRequested) {
        // The level-1 primitives and the factorizations are their own halves of the seam.
        BackendRegistry.registerAutomatic(dense.kernels)
        dense.decompositions.takeIf { it.isAvailable }?.let { BackendRegistry.registerAutomatic(it) }
    }
    val sparse = F64SparseBackends(automatic.klu, automatic.umfpack)
    registerIfOffered(sparse.klu, sparseRequested)
    registerIfOffered(sparse.umfpack, sparseRequested)
}

/**
 * Registers [backend] when its library loaded and the deployment did not pin a different one, then runs
 * [alsoRegister] for the halves that come with it.
 */
private fun registerIfOffered(backend: Backend, requested: String?, alsoRegister: () -> Unit = {}) {
    if (!backend.isAvailable) return
    if (requested != null && requested != backend.name) return
    BackendRegistry.registerAutomatic(backend)
    alsoRegister()
}

/** Instantiate all registered providers, dropping any whose construction fails. */
private fun loadProviders(): List<Backend> {
    val providers = ArrayList<Backend>()
    loadProviders(Backend::class.java, providers)
    // The dense service type was the original public SPI. Keep it while providers migrate to [Backend],
    // which also permits sparse-only add-ons such as UMFPACK.
    loadProviders(com.eignex.koblas.dense.F64LinearAlgebra::class.java, providers)
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
            break // a malformed registration poisons the iterator, keep what loaded so far
        }
    }
}

/**
 * Makes the candidate load its native path and produce a correct result.
 *
 * Sized from the level-3 gate, because a candidate that gates on these same thresholds answers anything
 * below its own gate from its portable fallback, where a missing library cannot show up. Every gate sits
 * above 1, and level 2 stays portable outright on a JVM carrying the Vector API. A level pinned portable
 * has no native path to reach at any size, so the probe stays at the cheap sanity check rather than
 * allocating for the pinning value.
 *
 * The primitive is called rather than the convenience overload, since a backend built by delegation
 * inherits a forwarder for every routine it does not override and the convenience one would bypass it.
 */
internal fun probe(backend: Backend): Boolean {
    if (backend !is F64Blas) return backend.isAvailable
    val gate = f64DispatchThresholds.level3
    val n = if (gate == Int.MAX_VALUE) 1 else maxOf(1, gate)
    @Suppress("TooGenericExceptionCaught") // native load failures surface as UnsatisfiedLinkError
    return try {
        val a = F64DenseMatrix(n, n, DoubleArray(n * n) { 2.0 })
        val c = F64DenseMatrix(n, n)
        backend.gemm(1.0, a, transposeA = false, a, transposeB = false, beta = 0.0, c = c)
        val expected = 4.0 * n
        c.data.all { it == expected }
    } catch (_: Throwable) {
        false
    }
}
