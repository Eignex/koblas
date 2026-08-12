package com.eignex.koblas

/**
 * Marks a test that needs a real host library (OpenBLAS, SuiteSparse). Excluded from the default run and
 * enabled with `-Pkoblas.hostTests=true`.
 */
interface HostLibraryTest
