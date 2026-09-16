# koblas-vendor-runtime

Optional vendor BLAS runtimes, packaged for Koblas. This module contains no Koblas code and no vendor binaries:
it downloads a pinned vendor package, checks it against a recorded hash, and lays the files out under the path
`com.eignex.koblas.vendor.Bundle` reads back.

Koblas does not need this artifact. A host with the vendor installed uses the installed library, which always
wins: the operator chose it, built or tuned it for the host, and the rest of their stack may already link it.
A payload is for a host that has none.

## Layout

```
com/eignex/koblas/vendor/<platform>/<vendor>/
    payload              the file names that make up the runtime, one per line, entry point first
    <libraries>          the runtime itself
    LICENSE.txt, ...     the vendor's own notices, never extracted at run time
```

One string addresses both runtimes. On the JVM it is a classpath resource, extracted to a private temporary
directory when no installed library answered. For a Native binary it is a directory on disk, looked for beside
the executable and in the working directory, and opened where it lies.

## What is published

| Vendor | Platform | Version | Published |
|---|---|---|---|
| oneMKL | linux-x86_64 | 2026.1.0-236 | yes |
| AOCL (BLIS) | linux-x86_64 | 5.3.0 | no, assembled locally |
| ArmPL | linux-arm64 | 25.07 | no, assembled locally |

Accelerate is part of macOS and is never packaged.

Only oneMKL is published, and the reason is licensing rather than anything technical. Every payload here is
built the same way and loads the same way.

- **oneMKL** is under the Intel Simplified Software License, which permits redistributing the binaries
  unmodified with the notice reproduced. That is what this artifact does.
- **AOCL** is under AMD's EULA, which permits distribution only as part of your own product and only to
  recipients who are contractually bound to that same EULA. A public artifact of the library alone is neither.
- **ArmPL** is under Arm's simplified EULA for free-of-charge redistributables, which permits redistribution
  "only as part of Your Software, provided Your Software contains substantial additional functionality", again
  with recipients bound to the same terms. An artifact carrying only the library does not meet that.

Both of those terms do allow an operator to build the payload into their own application, which is what the
`assemble…` tasks below are for. Reading the licence that ships in the payload directory is part of using one.

## Building a payload

```
./gradlew :koblas-vendor-runtime:jarLinuxX86_64          # the published artifact
./gradlew :koblas-vendor-runtime:assembleAoclLinuxX86_64 # local, not published
./gradlew :koblas-vendor-runtime:assembleArmplLinuxArm64 # local, not published
```

Each downloads its vendor package to `build/payload-downloads`, verifies the SHA-256 recorded in
`build.gradle.kts`, and keeps it for later builds. The oneMKL payload is 396 MB of libraries in a 122 MB jar:
all four instruction-set layers are packaged because `mkl_rt` opens the one matching the host at run time, and a
payload missing the host's layer does not fail, it quietly runs a narrower one.

Nothing here runs as part of `assemble` or `check`. The packaging needs `ar` and `tar`, which a Linux host
building a Linux payload has.

## Verifying one

```
./gradlew :koblas-vendor-runtime:verifyPayload           # JVM: extraction and loading
./gradlew :koblas:linuxX64Test -Pkoblas.nativePayload=true   # Native: the same layout on disk
```

Both run with `HOME` pointing at an empty directory so that no installed library can answer in the payload's
place, and both check that the library that answered came out of the payload, computed the right product, and
reported one compute thread.

## Updating a version

Change the URL, hash and paths in the `payloads` list in `build.gradle.kts`, then run the packaging and
verification above. The `payload` manifest is written from what was packaged, so it cannot claim a file the
build did not put beside it. A vendor whose entry-point file name changes also needs `Vendor.bundledFile`
updated in `:koblas`, which is what the loader looks for.
