# Published JVM consumers

These consumers resolve the same Maven publication from `koblas/build/staging-repo`. The library itself
has one JVM target; the Android application only tests the published JAR and packages no native code.

From the repository root, stage the fixed test version and run the desktop consumers:

```sh
./gradlew :koblas:publishJvmPublicationToLocalStagingRepository \
  :koblas:publishKotlinMultiplatformPublicationToLocalStagingRepository -PciVersion=0.0.0-consumer
./gradlew -p tests/consumers --refresh-dependencies :jvm:jvm17Consumer :jvm:jvm25Consumer \
  :jvm:jvm17ModuleConsumer :jvm:jvm25ModuleConsumer
```

With an Android SDK configured, build both ordinary D8 and shrinking R8 consumers at minSdk 24.
The test-report verifier requires `xmllint` (`libxml2-utils` on Ubuntu):

```sh
./gradlew -p tests/consumers :android:assembleDebug :android:assembleRelease
./gradlew -p tests/consumers :android:connectedDebugAndroidTest
./gradlew -p tests/consumers :android:connectedReleaseAndroidTest -Pconsumer.buildType=release
bash tests/verify-test-results.sh \
  tests/consumers/android/build/outputs/androidTest-results/connected/debug \
  tests/consumers/android/build/outputs/androidTest-results/connected/release
```

Device tests use Java method names so their test APK remains compatible with API 24 DEX. Run them on an
API 24 emulator as well as a current Android runtime. The shared check exercises repeated `J * x`, `J^T * r`,
and `J^T * J` products against a straightforward scalar reference, reused workspaces, and an aliased destination.
It exercises both the platform default and the exact `BuiltinEngines.scalar` engine, checking
repeated finite-data results against the reference with rounding tolerance. Engine selection and metadata
inspection are directly callable.

CI stages a unique version per workflow attempt and passes it with `-Pconsumer.version=VERSION`, so it
does not need to refresh all dependencies. One producer runs the JVM consumers, verifies the publication
and both Android APKs, and builds debug and release instrumentation APKs. One Android job downloads the
verified APK artifact and runs it on API 24 and 34 with `.github/scripts/run-android-consumers.sh` without
rebuilding the library or applications. Both runtimes execute even when one fails, and both must pass.
Each API level has its own emulator name, cached snapshot, and reports, scoped to the runner image.

The [iOS consumer](../ios-consumer/README.md) runs this same check on an iOS simulator and links both
device and simulator frameworks from the published Kotlin/Native dependencies.

The shared fixture also exercises sparse operands through explicit generic engine dispatch, padded dense
destinations and signed vector strides.
These checks run from the staged publication on JVM classpath/module-path and Android D8/R8 consumers.
