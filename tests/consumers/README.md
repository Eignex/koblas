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

With an Android SDK configured, build both ordinary D8 and shrinking R8 consumers at minSdk 24:

```sh
./gradlew -p tests/consumers :android:assembleDebug :android:assembleRelease
./gradlew -p tests/consumers :android:connectedDebugAndroidTest
./gradlew -p tests/consumers :android:connectedReleaseAndroidTest -Pconsumer.buildType=release
python3 tests/verify-android-tests.py \
  tests/consumers/android/build/outputs/androidTest-results/connected/debug \
  tests/consumers/android/build/outputs/androidTest-results/connected/release
```

Device tests use Java method names so their test APK remains compatible with API 24 DEX. Run them on an
API 24 emulator as well as a current Android runtime. The shared check exercises repeated `J * x`, `J^T * r`,
and `J^T * J` products against the scalar oracle, reused workspaces, and an aliased destination.
