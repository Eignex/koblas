package com.eignex.koblas.dense

/**
 * Whether an OpenBLAS reporting this `openblas_get_config` string takes 64-bit integers.
 *
 * Both koblas bindings declare every integer parameter of every routine as 32 bits, which is the default
 * LP64 build. An OpenBLAS built with `INTERFACE64=1` exports the *same unsuffixed symbol names* and takes
 * 64-bit integers, so it resolves, registers, and then reads the wrong halves of every dimension it is
 * handed — wrong answers or a segmentation fault, with nothing in the resolution path able to notice.
 *
 * A computation would catch it, and used to: the `1x1 gemv` probe both platforms once ran came back wrong
 * because the argument widths did not line up. The probes are gone, deliberately, so this replaces them with
 * something cheaper and narrower — one symbol lookup and one call returning a string, which is a lookup and
 * not arithmetic, so it does not reopen what the probes cost.
 *
 * Two markers, because OpenBLAS has spelled it both ways: current builds put `USE64BITINT` in the config
 * string, and some report `INTERFACE64`. Either is disqualifying. A build that offers no config string at
 * all cannot be checked, and is assumed to be LP64 — the same assumption koblas made before this existed.
 *
 * Split out into common code because it is the only part of the check that is neither JVM nor native, and
 * because a string predicate is testable where a `dlopen` of a hypothetical ILP64 library is not.
 */
internal fun isIlp64OpenBlas(config: String): Boolean =
    // Whole tokens, not substrings: the config string is space-separated flags, and a `contains` would read
    // a hypothetical USE64BITINT_OFF as the marker it is the negation of.
    config.split(' ', '\t', '\n').any { it == "USE64BITINT" || it == "INTERFACE64" }
