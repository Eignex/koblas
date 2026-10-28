# Published iOS consumer

This independent Kotlin Multiplatform build resolves staged Maven publications rather than project sources.
It shares the JVM and Android consumer's numerical check, runs it on an arm64 iOS simulator, and links
static frameworks for an arm64 device and simulator. Linking and simulator execution require macOS and Xcode.

From the repository root on macOS:

```sh
./gradlew :koblas:publishKotlinMultiplatformPublicationToLocalStagingRepository \
  :koblas:publishIosArm64PublicationToLocalStagingRepository \
  :koblas:publishIosSimulatorArm64PublicationToLocalStagingRepository -PciVersion=0.0.0-consumer
./gradlew -p tests/ios-consumer --refresh-dependencies iosSimulatorArm64Test \
  linkDebugFrameworkIosArm64 linkDebugFrameworkIosSimulatorArm64
python3 tests/verify-test-results.py tests/ios-consumer/build/test-results/iosSimulatorArm64Test
```

Pass `-PiosSimulatorDevice=<simulator-uuid>` to select an installed simulator. CI selects and boots an available
iPhone before running both the library conformance suite and this consumer. The consumer checks portable
execution without a vendor, repeated transpose products with reused buffers, bitwise repeatability and aliases.
