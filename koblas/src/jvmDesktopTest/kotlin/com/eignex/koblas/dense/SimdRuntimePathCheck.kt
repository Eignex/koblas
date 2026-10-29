package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.jvm25.hardwareFusedMultiplyAdd
import com.eignex.koblas.internal.numeric.assertHardwareFmaAgreesWithVm
import com.eignex.koblas.poisonedTriangle
import jdk.incubator.vector.DoubleVector
import kotlin.random.Random
import kotlin.test.assertEquals

/**
 * Runs the panel, product and diagonal substitution contracts on this JVM's species and multiply-add.
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
        assertHardwareFmaAgreesWithVm()
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
        assertReductionsAgreeWithReference(engine.vectorKernels)
        SimdOpsTest().`dot agrees with the scalar oracle across independent offsets and vector boundaries`()
        assertIamaxHonoursItsContract(engine.vectorKernels)
        SimdOpsTest().`iamax agrees with the scalar oracle when blocks keep improving`()
        SimdOpsTest().`iamax distinguishes magnitudes sharing a high word`()
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
        val triangles = engine.triangularKernels
        check(triangles.name == "simd-substitution($lanes lanes)") {
            "the backend named ${triangles.name} at $lanes lanes"
        }
        assertTriangularKernelsAgreeWithReference(triangles, lanes)
        println("panel, tile and substitution conformance passed at $lanes lanes with fma=$hardwareFusedMultiplyAdd")
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
        assertDepthBlocksAccumulate(products)
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

    /**
     * At least two steps are needed to reach multiply-subtract in a solve and multiply-add in a product.
     * Lane boundaries distinguish the vector body, its portable tail and the wholly portable fallback.
     */
    private fun assertTriangularKernelsAgreeWithReference(kernels: DenseTriangularKernels, lanes: Int) {
        val rng = Random(20261013)
        val order = 3
        val n = order + 2
        val start = 1
        val xOffset = 3
        for (sides in intArrayOf(lanes - 1, lanes, lanes + 1)) {
            for (rhsStride in intArrayOf(1, 2)) {
                val orderStride = sides * rhsStride + 2
                val bodies = when {
                    rhsStride != 1 || sides < lanes -> listOf(PortableTriangularKernels.name)
                    sides == lanes -> listOf(kernels.name)
                    else -> listOf(kernels.name, PortableTriangularKernels.name)
                }
                assertEquals(
                    bodies,
                    kernels.implementationsFor(order, sides, rhsStride == 1),
                    "diagonal block order=$order sides=$sides rhsStride=$rhsStride",
                )
                for (lower in booleanArrayOf(false, true)) {
                    for (transposed in booleanArrayOf(false, true)) {
                        for (unitDiag in booleanArrayOf(false, true)) {
                            val (triangle, _) = poisonedTriangle(rng, n, lower != transposed, unitDiag)
                            val initial = DoubleArray(xOffset + order * orderStride + 4) { -17.0 }
                            for (p in 0 until order) {
                                for (r in 0 until sides) {
                                    initial[xOffset + p * orderStride + r * rhsStride] = rng.nextDouble(-1.0, 1.0)
                                }
                            }
                            for (solve in booleanArrayOf(false, true)) {
                                val expected = initial.copyOf()
                                val actual = initial.copyOf()
                                for ((backend, x) in listOf(PortableTriangularKernels to expected, kernels to actual)) {
                                    if (solve) {
                                        backend.diagonalSolve(
                                            triangle.values, n, start, order, transposed, lower, unitDiag,
                                            x, xOffset, orderStride, rhsStride, sides,
                                        )
                                    } else {
                                        backend.diagonalMultiply(
                                            triangle.values, n, start, order, transposed, lower, unitDiag,
                                            x, xOffset, orderStride, rhsStride, sides,
                                        )
                                    }
                                }

                                assertClose(
                                    expected,
                                    actual,
                                    "diagonal solve=$solve lower=$lower transposed=$transposed unit=$unitDiag " +
                                        "sides=$sides rhsStride=$rhsStride bodies=$bodies",
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
