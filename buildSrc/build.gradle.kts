// Task types for the bundled Android OpenBLAS. They live here rather than in koblas/build.gradle.kts because a
// task class declared in a build script is reimplemented by every edit to that script, which reruns an
// OpenBLAS build that takes minutes and whose inputs did not change.
plugins {
    `kotlin-dsl`
}

repositories {
    mavenCentral()
}
