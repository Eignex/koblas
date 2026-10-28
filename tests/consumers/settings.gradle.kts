pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositories {
        maven { url = uri("../../koblas/build/staging-repo") }
        google()
        mavenCentral()
    }
}
rootProject.name = "koblas-consumers"
include(":jvm", ":android")
