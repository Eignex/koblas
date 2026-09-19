package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

/**
 * Runs the panel contract on whatever species and multiply-add this JVM was started with.
 *
 * The bodies branch on two properties of the machine: how many lanes the preferred species has, and whether
 * a fused multiply-add has an instruction behind it. A test task cannot vary either, because both are fixed
 * when the virtual machine starts. This runs as its own process so the `simdNarrowSpeciesCheck` and
 * `simdNoFmaCheck` tasks can start it with `-XX:MaxVectorSize` and `-XX:-UseFMA` and hold the same
 * conformance to a narrower species and to the unfused arithmetic.
 *
 * Forcing a flag is not the same as owning the hardware. What this establishes is that the generated paths
 * are exercised and correct under each configuration, not how they perform on a machine that has no such
 * instruction.
 */
internal object SimdRuntimePathCheck {
    @JvmStatic
    fun main(args: Array<String>) {
        val lanes = DoubleVector.SPECIES_PREFERRED.length()
        println("preferred lanes=$lanes hardware fma=$hardwareFusedMultiplyAdd")
        val expectedLanes = args.getOrNull(0)?.toIntOrNull()
        if (expectedLanes != null) {
            check(lanes == expectedLanes) { "this process resolved $lanes lanes, expected $expectedLanes" }
        }
        val expectedFma = args.getOrNull(1)?.toBooleanStrictOrNull()
        if (expectedFma != null) {
            check(hardwareFusedMultiplyAdd == expectedFma) {
                "this process reports fma=$hardwareFusedMultiplyAdd, expected $expectedFma"
            }
        }
        val engine = requireNotNull(BuiltinEngines.simd) { "no Vector API arm" }
        val panels = engine.panelKernels
        check(panels.name == "simd-panel($lanes lanes)") { "the backend named ${panels.name} at $lanes lanes" }

        assertPanelKernelsAgreeWithReference(panels)
        assertPanelContractHolds(panels)
        assertEmptyExtentsReadNothing(panels)
        assertPanelsStayInsideTheirWindows(panels)
        assertExecutionGroupIsUsable(panels)
        // The route as well as the arithmetic. Which bodies a schedule reaches moves with the species: a
        // symmetric traversal grouped by two cuts only even windows at an even order, so the same call is a
        // composition at four lanes and direct at two. Checking only the raw panels here would leave that to
        // whichever machine happened to run the ordinary tests.
        assertRouteNamesExecutedBodies(panels)

        // The eligibility follows the species this process resolved, not the one it was written against.
        check(panels.implementationFor(PanelWork.MultiDot, lanes, 4) == panels.name) {
            "a panel of one whole lane block did not reach the vector body at $lanes lanes"
        }
        check(panels.implementationFor(PanelWork.MultiDot, lanes - 1, 4) == PortablePanelKernels.name) {
            "a panel shorter than a lane block reached the vector body at $lanes lanes"
        }
        // Printed because it is the thing that moves with the species and is easy to assume instead of
        // reading: a symmetric traversal grouped by two cuts only even windows at an even order.
        for (order in intArrayOf(512, 513)) {
            val route = engine.denseRouteOf(DenseMatrixOperation.Symv, DenseCall(order, order))
            println("symv order $order at $lanes lanes: ${route.kind} ${route.components}")
        }
        checkProducts(engine, lanes)
        println("panel and tile conformance passed at $lanes lanes with fma=$hardwareFusedMultiplyAdd")
    }

    /**
     * The product tile at whatever species this process resolved, and the route over it.
     *
     * The tile's rows are lane blocks, so which destinations have a remainder and which do not moves with
     * the species: the same product is one body wide at one width and a composition at another. A check that
     * only ran the arithmetic would leave that to whichever machine happened to run the ordinary tests.
     */
    private fun checkProducts(engine: KoblasEngine, lanes: Int) {
        val products = engine.productKernels
        check(products.tileRows % lanes == 0) {
            "the product tile has ${products.tileRows} rows, which is not lane blocks at $lanes lanes"
        }
        assertProductBlockAgreesWithReference(products)
        assertEmptyProductBlockReadsNothing(products)
        assertZeroBetaOverwritesPoison(products)
        assertDepthSlicesAccumulate(products)
        for ((m, n, k) in listOf(
            Triple(products.tileRows * 4, products.tileColumns * 4, 64),
            Triple(products.tileRows * 4 + 1, products.tileColumns * 4 + 1, 64),
            Triple(37, 29, 41),
        )) {
            assertProductRouteNamesExecutedBlocks(products, engine.panelKernels, m, n, k)
            val route = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(m, n, depth = k))
            println("gemm ${m}x${n}x$k at $lanes lanes: ${route.kind} ${route.components}")
        }
    }
}
