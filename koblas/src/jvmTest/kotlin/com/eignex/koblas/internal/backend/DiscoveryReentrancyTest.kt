package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64VectorKernels
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A provider whose `gemv` runs its inner loop on the installed kernels, which is what a host backend does
 * and what the discovery probe calls. It does not override [F64Blas.vectorKernels], so that read resolves
 * through [koblas]: a read of the very value discovery is computing.
 */
public class ProbeReentrantProvider :
    F64LinearAlgebra,
    F64Blas by F64ReferenceLinearAlgebra,
    F64Lapack by F64ReferenceLinearAlgebra {

    init {
        instantiations.incrementAndGet()
    }

    override val name: String get() = "probe-reentrant"
    override val priority: Int get() = 200
    override val isPortable: Boolean get() = false
    override val isAvailable: Boolean get() = true

    /** What [F64LinearAlgebra] resolves to by default, spelled out because the delegations declare it too. */
    override val vectorKernels: F64VectorKernels get() = koblas.vectorKernels

    /**
     * What the probe actually calls. Spelled out because delegating [F64Blas] generates a forwarder for
     * every member including this one, which would send the probe to the delegate rather than here. A real
     * provider implements its own routines and inherits this default, which lands on the override below.
     */
    override fun gemv(a: F64DenseMatrix, x: DoubleArray, transpose: Boolean): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        val kernels = vectorKernels // the read that closes the loop back onto discovery
        y.fill(0.0)
        kernels.axpy(y, 0, alpha * x[0], a.data, 0, y.size)
    }

    /** How many of these the process has built, which is one per discovery pass. */
    public companion object {
        public val instantiations: AtomicInteger = AtomicInteger()
    }
}

/** Reads [koblas] on a JVM where [ProbeReentrantProvider] is service-loaded, and reports what happened. */
public fun main() {
    val resolved = koblas.name
    println("resolved=$resolved passes=${ProbeReentrantProvider.instantiations.get()}")
}

/**
 * Discovery probes every ServiceLoader provider by running a small gemv on it, and a provider that has not
 * overridden [F64Blas.vectorKernels] reads [koblas] to get them. Nothing stopped that read from restarting
 * discovery, so it recursed until the stack ran out.
 *
 * The provider has to arrive through a real [java.util.ServiceLoader] lookup in a fresh process, since
 * discovery runs once per JVM and this one has already run it. So the test forks one, which is why it is
 * the one JVM test over the 300ms budget.
 */
class DiscoveryReentrancyTest {

    @Test
    fun `a service-loaded provider does not restart discovery while being probed`() {
        val services = Files.createTempDirectory("koblas-discovery")
        val registrations = services.resolve("META-INF/services")
        Files.createDirectories(registrations)
        Files.writeString(
            registrations.resolve("com.eignex.koblas.dense.F64LinearAlgebra"),
            "com.eignex.koblas.internal.backend.ProbeReentrantProvider\n",
        )
        val separator = System.getProperty("path.separator")
        val process = ProcessBuilder(
            "${System.getProperty("java.home")}/bin/java",
            "--enable-native-access=ALL-UNNAMED",
            "--add-modules=jdk.incubator.vector",
            "-cp",
            "$services$separator${System.getProperty("java.class.path")}",
            "com.eignex.koblas.internal.backend.DiscoveryReentrancyTestKt",
        ).redirectErrorStream(true).start()
        val output = process.inputStream.bufferedReader().readText()
        val status = process.waitFor()
        val head = output.lineSequence().take(12).joinToString("\n")
        assertEquals(0, status, "the forked JVM failed:\n$head")
        val report = output.lineSequence().firstOrNull { it.startsWith("resolved=") }
        assertTrue(report != null, "the forked JVM never resolved a context:\n$head")
        // One pass. Before the guard this recursed until the stack ran out, thousands of passes deep, and
        // the StackOverflowError that ended it was swallowed by the probe's own catch.
        assertEquals("passes=1", report.substringAfter(' '), "discovery restarted while probing: $report")
        assertTrue("probe-reentrant" in report, "the probed provider should still register: $report")
    }
}
