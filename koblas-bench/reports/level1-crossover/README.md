# Level 1 crossover calibration

The measurements that chose `VendorVectorKernels.crossover` and confirmed `SimdVectorKernels.IAMAX_CROSSOVER`.
Unlike the hardware-keyed directories beside this one, this is not a default capture: it is nine sweep captures,
one per operation, concatenated per target so that one file holds one arm's whole answer.

Each capture ran

```
koblas-bench/capture-report.sh --libraries onemkl --suite sweep --operation <name> \
  --samples 5 --warmups 5 --target-ms 200 --forks 2
```

pinned with `taskset -c 2,4,6,8` to four P-cores. This host is a hybrid part whose E-cores run a gigahertz
slower, and an unpinned run wanders between the two and reads as a regression that is only scheduling.
`metadata.txt` is the `dot` capture's, which the other eight share except for their timestamps; it records the
resolved library file, its version, and the thread count the binding read back after holding it to one.

## Files

| File | Arm |
|---|---|
| `native.csv` | portable Kotlin on Kotlin/Native |
| `onemkl.csv` | oneMKL through the Native binding, which passes the caller's array in place |
| `jvm-scalar.csv` | portable Kotlin on the JVM |
| `jvm-simd.csv` | Vector API kernels on the JVM |
| `onemkl-jvm.csv` | oneMKL through the JVM binding, whose timing includes copying both operands |

`sum` has no vendor rows: it is not a BLAS routine, so the vendor arms report it unsupported rather than timing
a substitute.

## The crossover this chose

Portable time divided by vendor time on Kotlin/Native. Break-even is interpolated between the two swept widths
that bracket a ratio of 1.

| Operation | Break-even | 32 | 64 | 96 | 128 | 256 | Best |
|---|---|---|---|---|---|---|---|
| `axpy` | 40 | 0.87 | 1.30 | 1.55 | 1.83 | 2.88 | 2.9 |
| `scal` | 42 | 0.90 | 1.17 | 1.46 | 1.65 | 2.59 | 4.9 |
| `asum` | 45 | 0.80 | 1.26 | 1.67 | 1.83 | 3.28 | 8.6 |
| `rot` | 46 | 0.82 | 1.19 | 1.31 | 1.88 | 2.59 | 2.6 |
| `dot` | 57 | 0.65 | 1.17 | 1.60 | 1.78 | 2.45 | 4.9 |
| `swap` | 67 | 0.60 | 0.99 | 1.07 | 1.36 | 1.78 | 1.8 |
| `nrm2` | 82 | 0.64 | 0.89 | 1.07 | 1.08 | 1.38 | 2.1 |
| `iamax` | 100 | 0.33 | 0.66 | 0.89 | 1.99 | 3.55 | 9.4 |

Two constants, because the eight fall into two groups with a gap between them. The first six meet the library
where the portable loop's work grows to the size of a call: each vendor call costs a flat 33 to 60 ns whatever
the width, and the loop costs about half a nanosecond per element. `nrm2` and `iamax` arrive later for reasons
of their own — `dnrm2` rescales for overflow safety as it goes, so its cost grows with the width from the
start, and `idamax` charges 60 to 120 ns before it looks at anything, which is oneMKL's own number and not the
binding's: timed directly from C, `cblas_idamax` costs 89 ns over 64 entries where `cblas_dasum` costs 10.

## These numbers belong to a particular binding

An earlier binding spent about 190 ns per call on the objects that described each operand — a holder for the
pins, a wrapper per operand, a vector object per operand, and a list allocated to ask whether there was work.
Measured against it the same eight break-evens were 205 to 652: five times higher, in a different order, and
grouped differently, with `iamax` first rather than last. A crossover is a property of the call as much as of
the arithmetic, so a change to what a call costs invalidates these rather than shifting them.

The per-call cost now is 33 ns for one pinned operand and 42 ns for two. Of that, about 9.5 ns is each pin,
7.6 ns is the call itself as timed from C, and the rest is dispatch. Pinning is what it costs to hand a
GC-managed array to C; reading the same data through a pinned `CPointer` instead of a `DoubleArray` measured
five times slower per element, so there is no cheaper arrangement to move to.

## The JVM answers differently

`jvm-simd.csv` against `onemkl-jvm.csv` has the vendor behind at every width of every operation, by 6 to 23
times. Reaching the library from the JVM copies both operands into native memory, so the call pays a pass over
the data before it computes anything. Even against `onemkl.csv`, which copies nothing and is the best the JVM
could ever do, the Vector API kernels stay ahead into the hundreds of elements: for `asum` at 256, 18.4 ns
against 48.6. There is no JVM crossover to set.

## `jvm-iamax-gate-lowered/`

`SimdVectorKernels.IAMAX_CROSSOVER` gates its own measurement: with the shipped value of 256, every width below
it runs the scalar kernel in both arms and the comparison is of one kernel against itself. These two files come
from one capture with that constant temporarily set to the lane width, so the vectorised search runs from 4
upward and can be timed where it is normally not used. Nothing else differed, and the source was restored
afterwards.

It reads: 0.43 at 8, 0.54 at 16, 0.73 at 32, then 1.01 at 64, 1.13 at 96, 0.88 at 128, 1.02 at 192, and from 256
it is ahead at every width measured (1.14, 1.19, 1.13, 1.60, 1.59, 1.84, 1.90). Below 256 the vectorised search
is inside the noise and loses outright at one width; 256 is where it starts winning and staying ahead, so the
existing constant is kept, now with a measurement behind it rather than an estimate.
