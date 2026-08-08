package com.eignex.koblas

/**
 * Marks a test that needs a real host library — OpenBLAS, SuiteSparse — rather than koblas's portable
 * implementations. Excluded from the default run, enabled with `-Pkoblas.hostTests=true`: these measure the
 * machine as much as the library, and they are what makes discovery bind FFM handles under the coverage agent.
 */
interface HostLibraryTest
