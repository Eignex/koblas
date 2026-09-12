# Scaling follow-up from the OpenBLAS source

The local OpenBLAS source exposed another useful improvement after alignment peeling: advance the data
pointer through fixed 16-element blocks, so the main native loop uses base-plus-displacement addresses.
The retained implementation is ordinary nested C loops, with four new lines in the existing aligned path.
It does not copy OpenBLAS assembly or introduce software pipelining.

Through the native engine, n=4096 arithmetic improves from 205.9 to 179.9 ns of thread CPU time for aligned
inputs (median paired 1.14x, range 1.13–1.16x). Misaligned inputs improve from 195.1 to 176.2 ns (1.11x,
range 1.10–1.11x). These are incremental gains over the alignment-only implementation, not over the original
unaligned implementation. Other measured native sizes are effectively unchanged; the misaligned n=256
paired median is 0.98x, and aligned n=65536 reset timing is 0.99x. No universal speedup is claimed.

## Source findings

The checkout is `/home/rasmus/Workspaces/OpenBLAs`, commit
`632ef874379c16ca8646d9fa0f6c60449e47d960`, with no local changes. The loaded library identifies itself as
`OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=128`; its selected core is Haswell.
The local Haswell microkernel and `kernel/x86_64/dscal.c` are unchanged from tag `v0.3.32`. The interface
has only an added `OPENBLAS_EXPORT` annotation relative to that tag.

- [The Haswell microkernel](https://github.com/OpenMathLib/OpenBLAs/blob/632ef874379c16ca8646d9fa0f6c60449e47d960/kernel/x86_64/dscal_microk_haswell-2.c#L48)
  seeds four YMM products, then interleaves their stores with the next four loads/multiplies. Its loop
  advances the pointer by 128 bytes and addresses memory by fixed displacements. The prefetch lines
  are commented out. Its stores accept unaligned addresses, and it does not peel a prefix for alignment.
- [The architecture selection](https://github.com/OpenMathLib/OpenBLAs/blob/632ef874379c16ca8646d9fa0f6c60449e47d960/kernel/x86_64/dscal.c#L28)
  chooses that microkernel for Haswell and Zen. This matches the installed library's runtime identity.
- [The public interface](https://github.com/OpenMathLib/OpenBLAs/blob/632ef874379c16ca8646d9fa0f6c60449e47d960/interface/scal.c#L79)
  returns for alpha=1 and keeps calls of at most 1,048,576 elements on one thread. The timings here are
  also explicitly limited to one vendor thread.

The existing Koblas AVX2 loop already handled four vectors per iteration, but its main stores used a
base plus scaled index. The fixed-block loop makes Clang emit base-plus-displacement stores, matching
one property of the OpenBLAS loop. This correlates with the improvement; without hardware counters the
experiment does not isolate address-generation resources from instruction scheduling and loop layout.

## Experiments and retained change

`probe.c` compares the alignment-only Koblas C leaf with an explicit vector/pointer loop, a C software
pipeline, the actual OpenBLAS Haswell microkernel, that microkernel with alignment peeling, and the two
public vendor interfaces. It includes the microkernel from the local checkout; it does not replace the
installed OpenBLAS used by the public vendor arm.

At aligned n=4096 with alternating 0.5 and 2.0, the pointer prototype takes 164.1 ns versus 195.0 ns for
the alignment-only leaf, 172.4 ns for the software pipeline, 165.1 ns for OpenBLAS, and 164.8 ns for
oneMKL. At an eight-byte offset, adding alignment to OpenBLAS's own leaf lowers its time from 397.1 to
162.5 ns. This independently reinforces the earlier alignment diagnosis.

`blocks.c` adds a simpler prototype: an outer pointer loop over 16-element blocks and a constant-bound
inner scalar loop. At aligned n=4096 it takes 166.5 ns versus 190.0 ns for alignment-only, 162.3 ns for
OpenBLAS, and 160.1 ns for oneMKL. This is close to the explicit-vector prototype while keeping the
production source simple. The production version retains the existing aligned-pointer assumption and
scalar tail; the actual native binary, not just the C prototype, supplies the before/after validation.

The native AVX2 assembly confirms fixed-displacement aligned stores in the main block loop. The x86 guard,
128-element cutoff, alpha=1 fast return and multiplication semantics remain unchanged. JVM scaling and
other architectures are unchanged. Conformance coverage adds tails around the first two 16-element
blocks after the cutoff.

## Measurement and reproduction

The standard suite's n=4096 arithmetic median improves from 224.9 ns for the original implementation
to 206.6 ns with both changes, versus 200.9 ns for oneMKL and 196.9 ns for OpenBLAS. At n=65536 it is
7,367.6 ns versus oneMKL's 7,367.4 ns and OpenBLAS's 8,106.0 ns. The 4096-element reset workload remains
slower than both vendors (801.0 ns versus 768.2 ns and 734.1 ns). Array alignment is uncontrolled in
these standard-suite runs, so the controlled native diagnostic remains the evidence for offset effects.

The [tables](tables.md) separate CPU-time diagnostics from the standard suite's elapsed-time comparison.
Both diagnostics use the same fixtures, clocks, offsets, warmup rules, pinned CPU 4, and alternating
passes as the [alignment report](../20260912T134906Z-scal-alignment/README.md). The first C probe rotates
seven arms; the block probe rotates eight, each for 32 rounds per sample, two warmup samples and seven
measured samples. Both negate and alternating exact powers of two preserve finite magnitudes. Reset
mode copies the original vector and multiplies by 0.875. Every raw sample is retained.

The native diagnostic has three warmup samples, seven measured samples, and three process passes in
alternating order. Each sample contains 16 batches of 1,048,576/n operations and records actual pointer
alignment. It calls the native engine with the canonical uniform fixture and consumes the first/last
results. Baseline and candidate use identical diagnostic source.

- Diagnostic baseline production source: `ef35f7ca6871efddc43194c55b0d92b63c42af88` (alignment only).
- Candidate production source: `10128c807d7b21990a9a043b34709d4c8f3d839e` (alignment plus fixed blocks).
- The baseline diagnostic binary is the after binary retained from the preceding alignment experiment.
- Standard-suite old native binary: `5e5a7f1f1de41b1824299822f66314b6af1574ef`, before either scaling change.
  The suite therefore compares the original implementation with the combined fix, whereas the controlled
  native diagnostic isolates the incremental block-loop change.
- Standard-suite new native binary and vendor harness: `10128c807d7b21990a9a043b34709d4c8f3d839e`.
  Raw rows contain source IDs, dirty state and harness settings. `suite/cpu.csv` records host load.

Run `bash reproduce-diagnostics.sh NEW_OUTPUT_DIRECTORY` from the Koblas checkout to build the two
production revisions with the same temporary diagnostic entry point. The C probes require the recorded
OpenBLAS checkout at the absolute include path in their source; its relevant file hashes are in
`metadata.txt`. Build against the baseline header, since the `current` C arm means alignment only.
The script cleans up its own temporary worktrees. `capture-suite.sh` preserves the standard-suite
commands and identifies the retained old binary by its /tmp path; adjust that path when reproducing.
Run `python3 summarize-results.py` to regenerate the tables without changing raw samples.

## Validation

Passed `./gradlew :koblas:check :koblas-bench:check lintDocs check` and
`./gradlew :koblas:jvmTest -Pkoblas.noSimd=true`. The scalar-oracle scaling test covers offsets, cutoff and
block boundaries, untouched surrounding storage, exceptional values and tails; its JVM time was 75 ms.
No changes were made to the OpenBLAS checkout. Its license accompanies the diagnostic disassembly.
