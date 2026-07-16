plugins {
    // Declared here with apply=false so subprojects don't each load the Kotlin plugin into
    // separate classloaders (which conflicts on shared build services like the KotlinNativeBundle).
    kotlin("multiplatform") version "2.3.20" apply false
}
