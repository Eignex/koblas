# Implementation inventory for issue 666

Baseline: `210a8b2e` (clean working tree). Counts include Kotlin and Java source files, excluding generated build output.

| Module | Production files | Production lines | Test files | Test lines |
| --- | ---: | ---: | ---: | ---: |
| koblas | 91 | 16144 | 118 | 21918 |
| koblas-bench | 26 | 3663 | 18 | 1897 |

The baseline compiles `commonTest` twice, in `test` and `portableTest`. Both `jvmTest` and `jvmTestJarOnJava25` execute the desktop packaged suite. Java 17 executes `portableTest`. `jvmModule` is a descriptor input rather than a source set.

## Baseline configured graph

```text
MODULE :koblas sourceSets=32
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE iosArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosArm64Main/kotlin]
SOURCE iosArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosArm64Test/kotlin]
SOURCE iosMain dependsOn=[appleMain] dirs=[src/iosMain/kotlin]
SOURCE iosSimulatorArm64Main dependsOn=[iosMain, posixMain] dirs=[src/iosSimulatorArm64Main/kotlin]
SOURCE iosSimulatorArm64Test dependsOn=[iosTest, posixTest] dirs=[src/iosSimulatorArm64Test/kotlin]
SOURCE iosTest dependsOn=[appleTest] dirs=[src/iosTest/kotlin]
SOURCE jvmJava25 dependsOn=[] dirs=[src/jvmJava25/java, src/jvmJava25/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmPortableTest dependsOn=[] dirs=[src/commonTest/kotlin, src/jvmPortableTest/java, src/jvmPortableTest/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain, posixMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest, posixTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain, posixMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest, posixTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
SOURCE posixMain dependsOn=[nativeMain] dirs=[src/posixMain/kotlin]
SOURCE posixTest dependsOn=[nativeTest] dirs=[src/posixTest/kotlin]
COMPILATION java25 sources=[jvmJava25] associates=[main]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION portableTest sources=[jvmPortableTest] associates=[main]
COMPILATION test sources=[commonTest, jvmTest] associates=[java25, main]
MODULE :koblas-bench sourceSets=22
SOURCE appleMain dependsOn=[nativeMain] dirs=[src/appleMain/kotlin]
SOURCE appleTest dependsOn=[nativeTest] dirs=[src/appleTest/kotlin]
SOURCE commonMain dependsOn=[] dirs=[src/commonMain/kotlin]
SOURCE commonTest dependsOn=[] dirs=[src/commonTest/kotlin]
SOURCE jvmMain dependsOn=[commonMain] dirs=[src/jvmMain/java, src/jvmMain/kotlin]
SOURCE jvmTest dependsOn=[commonTest] dirs=[src/jvmTest/java, src/jvmTest/kotlin]
SOURCE linuxArm64Main dependsOn=[linuxMain] dirs=[src/linuxArm64Main/kotlin]
SOURCE linuxArm64Test dependsOn=[linuxTest] dirs=[src/linuxArm64Test/kotlin]
SOURCE linuxMain dependsOn=[nativeMain] dirs=[src/linuxMain/kotlin]
SOURCE linuxTest dependsOn=[nativeTest] dirs=[src/linuxTest/kotlin]
SOURCE linuxX64Main dependsOn=[linuxMain] dirs=[src/linuxX64Main/kotlin]
SOURCE linuxX64Test dependsOn=[linuxTest] dirs=[src/linuxX64Test/kotlin]
SOURCE macosArm64Main dependsOn=[macosMain] dirs=[src/macosArm64Main/kotlin]
SOURCE macosArm64Test dependsOn=[macosTest] dirs=[src/macosArm64Test/kotlin]
SOURCE macosMain dependsOn=[appleMain] dirs=[src/macosMain/kotlin]
SOURCE macosTest dependsOn=[appleTest] dirs=[src/macosTest/kotlin]
SOURCE mingwMain dependsOn=[nativeMain] dirs=[src/mingwMain/kotlin]
SOURCE mingwTest dependsOn=[nativeTest] dirs=[src/mingwTest/kotlin]
SOURCE mingwX64Main dependsOn=[mingwMain] dirs=[src/mingwX64Main/kotlin]
SOURCE mingwX64Test dependsOn=[mingwTest] dirs=[src/mingwX64Test/kotlin]
SOURCE nativeMain dependsOn=[commonMain] dirs=[src/nativeMain/kotlin]
SOURCE nativeTest dependsOn=[commonTest] dirs=[src/nativeTest/kotlin]
COMPILATION main sources=[commonMain, jvmMain] associates=[]
COMPILATION test sources=[commonTest, jvmTest] associates=[main]
```
