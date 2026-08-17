package com.eignex.koblas.bench

import kotlin.random.Random

/** Every operand derives from this, so a re-run measures the same numbers rather than similar ones. */
internal const val BENCH_SEED = 20260730

/** The backend parameter value that leaves resolution to discovery. */
internal const val AUTO_BACKEND = "auto"

/** The backend parameter value that pins a run to the portable kernels. */
internal const val REFERENCE_BACKEND = "reference"

/** The kernels parameter value that clears the platform's level-1 kernels. */
internal const val BUILTIN_KERNELS = "builtin"

/** The kernels parameter value that installs them. */
internal const val HOST_KERNELS = "host"

/** The shape parameter value that takes a simplex-like basis. */
internal const val BASIS_SHAPE = "basis"

/** The shape parameter value that takes a matrix with uniformly scattered fill. */
internal const val RANDOM_SHAPE = "random"

/** Off-diagonal fill fraction for the sparse operands, low enough that the sparse paths stay sparse. */
internal const val SPARSE_DENSITY = 0.01

/** A scale just off one, so repeated in-place updates neither fold away nor drift out of range. */
internal const val NEAR_UNIT_SCALE = 1.000001

/** The generator every fixture draws from, seeded so all suites see the same operands. */
internal fun benchRng(): Random = Random(BENCH_SEED)
