package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.BundledNativeResources
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLu
import java.nio.file.Path

/** BASICLU bundle extracted from Maven-native resources on the application's classpath. */
public class BundledBasiclu private constructor(private val delegate: BasicluSparseLu) : F64SparseLu by delegate {
    /** Extracts the matching Maven-native resources before the FFM binding is initialized. */
    public constructor() : this(BasicluSparseLu(BasicluResources.extract().toString()))

    override val name: String get() = "basiclu-bundled"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 3

    /** Factor a simplex [basis] for sparse column replacements. */
    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization = delegate.factorBasis(basis)
}

internal object BasicluResources {
    private val platform = BundledNativeResources.supportedPlatform { _, _ ->
        "koblas-basiclu has no bundled BASICLU for this host"
    }
    private val resources = BundledNativeResources(
        directoryPrefix = "koblas-basiclu",
        platform = platform,
        resourceRoot = "org/eignex/basiclu",
        anchor = BundledBasiclu::class.java,
        libraryDescription = "BASICLU",
    )
    private val library: String = if (platform.startsWith("linux")) {
        "libkoblas_basiclu.so.1"
    } else {
        "libkoblas_basiclu.1.dylib"
    }
    private val extracted: Path by lazy {
        checkNotNull(resources.extractRequired(resourceNames())[library]) { "BASICLU resource is absent for $platform" }
    }

    fun extract(): Path = extracted

    private fun resourceNames(): List<String> = checkNotNull(resources.resource(".libraries")) {
        "BASICLU resources are absent for $platform"
    }.bufferedReader().useLines { it.filter(String::isNotBlank).toList() }
}
