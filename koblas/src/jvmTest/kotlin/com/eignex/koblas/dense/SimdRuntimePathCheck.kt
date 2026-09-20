package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector

/**
 * Runs the panel contract on whatever species and multiply-add this JVM was started with.
 *
 * Both are fixed when the virtual machine starts, so a test task cannot vary them. This runs as its own
 * process for the `simdNarrowSpeciesCheck` and `simdNoFmaCheck` tasks to start with `-XX:MaxVectorSize` and
 * `-XX:-UseFMA`. Forcing a flag establishes that the generated paths are correct, not how they perform.
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
        // Which bodies a schedule reaches moves with the species, so the same call is a composition at four
        // lanes and direct at two.
        assertRouteNamesExecutedBodies(panels)

        // The eligibility follows the species this process resolved, not the one it was written against.
        check(panels.implementationFor(PanelWork.MultiDot, lanes, 4) == panels.name) {
            "a panel of one whole lane block did not reach the vector body at $lanes lanes"
        }
        check(panels.implementationFor(PanelWork.MultiDot, lanes - 1, 4) == PortablePanelKernels.name) {
            "a panel shorter than a lane block reached the vector body at $lanes lanes"
        }
        // Printed because it is what moves with the species and is easy to assume instead of reading.
        for (order in intArrayOf(512, 513)) {
            val route = engine.routeOf(DenseMatrixOperation.Symv, DenseCall(order, order))
            println("symv order $order at $lanes lanes: ${route.kind} ${route.components}")
        }
        checkProducts(engine, lanes)
        println("panel and tile conformance passed at $lanes lanes with fma=$hardwareFusedMultiplyAdd")
    }

    /**
     * The product tile at whatever species this process resolved, and the route over it. The tile's rows are
     * lane blocks, so which destinations have a remainder moves with the species.
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
            val route = engine.routeOf(DenseMatrixOperation.Gemm, DenseCall(m, n, depth = k))
            println("gemm ${m}x${n}x$k at $lanes lanes: ${route.kind} ${route.components}")
        }
    }
}
