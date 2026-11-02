package com.eignex.koblas.internal.dense

/** The part of a square destination a selected-triangle routine writes. */
internal fun selectedTriangle(lower: Boolean): OutputTriangle =
    if (lower) OutputTriangle.Lower else OutputTriangle.Upper
