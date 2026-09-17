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

| Operation | Break-even | 256 | 384 | 512 | 768 | Best |
|---|---|---|---|---|---|---|
| `iamax` | 205 | 1.22 | 1.60 | 1.91 | 2.71 | 9.38 |
| `rot` | 236 | 1.06 | 1.35 | 1.59 | 2.20 | 2.53 |
| `scal` | 331 | 0.82 | 1.12 | 1.42 | 1.92 | 4.91 |
| `dot` | 354 | 0.76 | 1.07 | 1.24 | 1.72 | 4.26 |
| `asum` | 365 | 0.76 | 1.04 | 1.36 | 1.88 | 8.62 |
| `axpy` | 373 | 0.77 | 1.02 | 1.22 | 1.58 | 2.26 |
| `swap` | 403 | 0.75 | 0.97 | 1.16 | 1.42 | 1.65 |
| `nrm2` | 652 | 0.58 | 0.74 | 0.88 | 1.09 | 2.06 |

Three groups, because the ends do not overlap the middle and each has a reason: `iamax` and `rot` do several
operations per element in the portable loop, `nrm2` is the one case where the library does the extra work
(`dnrm2` rescales for overflow safety, where the portable kernel tries the plain sum of squares first), and the
remaining five stream one arithmetic step per element.

The elementwise operations give their advantage back at the widest widths: `swap` and `rot` return to 1.03 at
262144, where both arms are bound by memory rather than by arithmetic. That is a reason to read the whole curve
rather than the largest width alone, not a reason to route differently, since routing follows the width where
the call is repaid.

## The JVM answers differently

`jvm-simd.csv` against `onemkl-jvm.csv` has the vendor behind at every width of every operation. Reaching the
library from the JVM copies both operands into native memory, so the call pays a pass over the data before it
computes anything, and no width repays it. There is no JVM crossover to set.

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
