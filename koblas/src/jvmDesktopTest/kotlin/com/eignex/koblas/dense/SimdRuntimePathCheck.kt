package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.PortableTriangularKernels
import com.eignex.koblas.internal.jvm25.simd.SIMD_SPECIES
import com.eignex.koblas.internal.jvm25.simd.hardwareFusedMultiplyAdd
import com.eignex.koblas.internal.numeric.assertHardwareFmaAgreesWithVm
import com.eignex.koblas.poisonedTriangle
import kotlin.random.Random
import kotlin.test.assertEquals

private val selectedBackend = BuiltinEngines.simd ?: PortableEngine

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
        val lanes = SIMD_SPECIES.length()
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
        assertReductionsAgreeWithReference(engine)
        SimdOpsTest().`dot agrees with the scalar oracle across independent offsets and vector boundaries`()
        assertIamaxHonoursItsContract(engine)
        SimdOpsTest().`iamax agrees with the scalar oracle when blocks keep improving`()
        SimdOpsTest().`iamax distinguishes magnitudes sharing a high word`()
        val panels = engine
        check(
            panels.panelImplementation.name == "simd-panel($lanes lanes)",
        ) { "the backend named ${panels.name} at $lanes lanes" }

        assertPanelKernelsAgreeWithReference(panels)
        assertPanelContractHolds(panels)
        assertEmptyExtentsReadNothing(panels)
        assertPanelsStayInsideTheirWindows(panels)
        assertExecutionGroupIsUsable(panels)

        // The eligibility follows the species this process resolved, not the one it was written against.
        check(panels.implementationFor(PanelWork.MultiDot, lanes, 4) == panels.panelImplementation.name) {
            "a panel of one whole lane block did not reach the vector body at $lanes lanes"
        }
        check(panels.implementationFor(PanelWork.RankUpdate, lanes - 1, 4) == PortablePanelKernels.name) {
            "a rank update shorter than a lane block reached the vector body at $lanes lanes"
        }
        check(panels.implementationFor(PanelWork.MultiDot, lanes - 1, 4) == panels.panelImplementation.name) {
            "a reduction shorter than a lane block missed its short body at $lanes lanes"
        }
        checkProducts(engine, lanes)
        val triangles = engine
        check(triangles.triangularImplementation.name == "simd-substitution($lanes lanes)") {
            "the backend named ${triangles.name} at $lanes lanes"
        }
        assertTriangularKernelsAgreeWithReference(triangles, lanes)
        assertSingleSideAgreesWithReference(triangles, lanes)
        assertSparseDotAgreesWithReference()
        println("panel, tile and substitution conformance passed at $lanes lanes with fma=$hardwareFusedMultiplyAdd")
    }

    /**
     * The product tile at whatever species this process resolved. The tile's rows are lane blocks, so which
     * destinations have a remainder moves with the species.
     */
    private fun checkProducts(engine: KoblasEngine, lanes: Int) {
        val products = engine
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
            // Printed because it is what moves with the species and is easy to assume instead of reading.
            println("gemm ${m}x${n}x$k at $lanes lanes: ${products.implementationsFor(m, n, k)}")
        }
    }

    /**
     * At least two steps are needed to reach multiply-subtract in a solve and multiply-add in a product.
     * Lane boundaries distinguish the vector body, its portable tail and the wholly portable fallback.
     */
    private fun assertTriangularKernelsAgreeWithReference(kernels: KoblasEngine, lanes: Int) {
        val rng = Random(20261013)
        val order = 3
        val n = order + 2
        val start = 1
        val xOffset = 3
        // Four lane blocks go through each step together; five and one more also reach one block alone and the tail.
        for (sides in intArrayOf(lanes - 1, lanes, lanes + 1, 4 * lanes, 5 * lanes + 1)) {
            for (rhsStride in intArrayOf(1, 2)) {
                val orderStride = sides * rhsStride + 2
                val bodies = when {
                    rhsStride != 1 || sides < lanes -> listOf(PortableTriangularKernels.name)
                    sides % lanes == 0 -> listOf(kernels.triangularImplementation.name)
                    else -> listOf(kernels.triangularImplementation.name, PortableTriangularKernels.name)
                }
                assertEquals(
                    bodies,
                    kernels.implementationsFor(order, sides, if (sides == 1) orderStride == 1 else rhsStride == 1),
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

    /**
     * One contiguous side runs along the block. The oracle is the portable loop over every side, reached by
     * giving it a second side it does not otherwise take, so the new body is not checked against itself.
     */
    private fun assertSingleSideAgreesWithReference(kernels: KoblasEngine, lanes: Int) {
        val rng = Random(20261031)
        for (order in intArrayOf(lanes + 1, 4 * lanes + 3)) {
            val n = order + 2
            val start = 1
            check(
                kernels.implementationsFor(order, 1, contiguous = true) ==
                    listOf(kernels.triangularImplementation.name),
            ) {
                "a single contiguous side of order $order missed its body at $lanes lanes"
            }
            for (lower in booleanArrayOf(false, true)) {
                for (transposed in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (triangle, _) = poisonedTriangle(rng, n, lower != transposed, unitDiag)
                        val values = DoubleArray(order) { rng.nextDouble(-1.0, 1.0) }
                        for (solve in booleanArrayOf(false, true)) {
                            val paired = DoubleArray(2 * order) { if (it % 2 == 0) values[it / 2] else 0.0 }
                            val actual = values.copyOf()
                            if (solve) {
                                PortableTriangularKernels.diagonalSolve(
                                    triangle.values, n, start, order, transposed, lower, unitDiag, paired, 0, 2, 1, 2,
                                )
                                kernels.diagonalSolve(
                                    triangle.values, n, start, order, transposed, lower, unitDiag, actual, 0, 1, 1, 1,
                                )
                            } else {
                                PortableTriangularKernels.diagonalMultiply(
                                    triangle.values, n, start, order, transposed, lower, unitDiag, paired, 0, 2, 1, 2,
                                )
                                kernels.diagonalMultiply(
                                    triangle.values, n, start, order, transposed, lower, unitDiag, actual, 0, 1, 1, 1,
                                )
                            }

                            assertClose(
                                DoubleArray(order) { paired[2 * it] },
                                actual,
                                "single side order=$order solve=$solve lower=$lower transposed=$transposed " +
                                    "unit=$unitDiag",
                            )
                        }
                    }
                }
            }
        }
    }

    /** The exact indexed composition, whose vector body only two lanes reach, against the scalar loop. */
    private fun assertSparseDotAgreesWithReference() {
        val rng = Random(20261101)
        for (count in intArrayOf(1, 2, 3, 7, 32, 33, 257)) {
            val indices = IntArray(count + 2) { 3 * it + 1 }
            val values = DoubleArray(count + 3) { rng.nextDouble(-1.0, 1.0) }
            val dense = DoubleArray(3 * count + 8) { rng.nextDouble(-1.0, 1.0) }

            val expected = PortableEngine.dotDense(indices, 1, values, 2, count, dense)
            val actual = selectedBackend.dotDense(indices, 1, values, 2, count, dense)

            assertClose(expected, actual, "sparse dot count=$count engine=${selectedBackend.name}")
        }
    }
}
