# Native kernels

`kernels/koblas_probe.h` and `kernels/koblas_kernels.h` are declaration-only ABI headers. The shared build script
compiles `ordinary.c` into distinct scalar, SSE2/NEON, and AVX2 objects. Scalar disables both loop and SLP
vectorization. Probe/entry wrappers always use baseline flags, no LTO, and no fast-math or contraction flags.
Indexed kernels have a single scalar implementation; hidden compiler clones and IFUNC resolvers are removed.

The same pipeline builds the JVM shared library and Kotlin/Native static archives. Native tasks use the
selected Kotlin/Native distribution's compiler, archiver, target and sysroot. Compiler/archiver files,
configuration, target flags, script and all native sources are Gradle inputs. Shared-library builds record
compiler version and target as inputs. `toolchain.txt` records the actual compiler, target and flags beside
objects, outside JVM resources. macOS cross artifacts built without an Apple toolchain retain Kotlin scalar.
A compiler targeting another architecture/OS is rejected. Static libraries are embedded through cinterop's
`staticLibraries` and target-specific `-libraryPath`, never a host archive reused across targets.

The probe uses 4-byte-aligned, sized/versioned records of fixed-width words. HOST reports hardware facts,
OS usability and compiled features independently. KERNEL enumerates the static catalog or looks up an exact
ID; THREAD queries current lengths without changing permissions or vector lengths. Header comments define
all status, geometry, layout, addressing, numerical and state tags. Unknown feature bits survive Kotlin
decoding; no synthetic descriptor can authorize an actual C call. Process permission, thread preparation and
per-call state are distinct. No current backend needs preparation, so no preparation entry point is exported.

x86 discovery checks CPUID and only executes XGETBV after XSAVE/OSXSAVE support is established. YMM/ZMM state
must be enabled before the corresponding OS feature is reported. Arm hardware facts are explicitly marked
as OS-advertised: Linux uses [HWCAP](https://www.kernel.org/doc/html/latest/arch/arm64/elf_hwcaps.html), and macOS
uses exact [feature queries](https://developer.apple.com/documentation/apple-silicon/addressing-architectural-differences-in-your-macos-code).
SVE and SME lengths use read-only Linux queries when advertised; absent or failed queries remain unknown.
No signal probing, chip-name feature inference, permission mutation, or SME arithmetic is used.

Every typed entry validates its operation-specific ID and ISA eligibility before operands or results are
mutated. Callers retain responsibility for valid logical windows and permitted aliases, as in the existing
Kotlin kernel contracts. The JVM probe uses an ordinary FFM transition with native request/result buffers.
Compute keeps the existing heap-backed critical-call mechanism, with no retained pointers; Native pins arrays
for a call and decodes the same result words. Raw exact selection bypasses performance thresholds, preserving
semantic early exits. Normal engine policy retains its Kotlin small-call paths.

Run `:koblas:checkNativeKernels` for C ABI sizing/error/no-read tests and exact exported-symbol checks. The
common tests exercise each available ordinary variant against the scalar oracle and exercise synthetic future
ACE/AVX10/type/state cases without executing them. Cross compilation and emulated execution do not constitute
real target hardware evidence or performance calibration.
