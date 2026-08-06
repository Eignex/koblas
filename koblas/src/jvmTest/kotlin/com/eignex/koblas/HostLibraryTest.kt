package com.eignex.koblas

/**
 * Marks a test that needs a real host library — OpenBLAS, SuiteSparse — rather than koblas's portable
 * implementations.
 *
 * Excluded from the default `jvmTest` run, and enabled on demand with `-Pkoblas.hostTests=true`. Two
 * separate reasons, both worth stating because either alone would justify it:
 *
 * A host-library test measures the machine as much as the library. It passes, fails, or skips depending on
 * what happens to be installed, so a default suite that includes it gives different answers on a developer
 * box with SuiteSparse and a CI runner without it. Keeping it out of the default run makes the everyday
 * result mean one thing.
 *
 * And it is what triggers the instrumentation failure recorded in the acceleration notes: platform discovery
 * initializing the `java.lang.foreign` downcall handles makes the coverage agent's transformer throw, which
 * floods the log and, when a test fails, can take the test worker down with it. Running the default suite
 * host-free avoids that entirely; the on-demand run accepts the noise in exchange for exercising the
 * bindings.
 *
 * A JUnit 4 category rather than a JUnit 5 `@Tag`, because that is what this source set runs on
 * (`kotlin-test-junit` brings JUnit 4.13.2). It is also necessarily JVM-only: `commonTest` has no JUnit
 * annotations available. That costs nothing here, since a host-library test is inherently platform-specific
 * — the native equivalents already live in their own `hostBlasTest` source set.
 */
interface HostLibraryTest
