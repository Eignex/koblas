package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
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
        val panels = requireNotNull(BuiltinEngines.simd) { "no Vector API panel candidate" }.panelKernels
        check(panels.name == "simd-panel($lanes lanes)") { "the backend named ${panels.name} at $lanes lanes" }

        assertPanelKernelsAgreeWithReference(panels)
        assertPanelContractHolds(panels)
        assertEmptyExtentsReadNothing(panels)
        assertPanelsStayInsideTheirWindows(panels)
        assertExecutionGroupIsUsable(panels)

        // The eligibility follows the species this process resolved, not the one it was written against.
        check(panels.implementationFor(PanelWork.MultiDot, lanes, 4) == panels.name) {
            "a panel of one whole lane block did not reach the vector body at $lanes lanes"
        }
        check(panels.implementationFor(PanelWork.MultiDot, lanes - 1, 4) == PortablePanelKernels.name) {
            "a panel shorter than a lane block reached the vector body at $lanes lanes"
        }
        println("panel conformance passed at $lanes lanes with fma=$hardwareFusedMultiplyAdd")
    }
}
