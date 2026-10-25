rootProject.name = "koblas"

pluginManagement {
    repositories {
        maven("https://central.sonatype.com/repository/maven-snapshots/") {
            mavenContent { snapshotsOnly() }
            content { includeGroupByRegex("com\\.eignex.*") }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    // Provisions the JDK the conventions ask for, so a machine without it does not need one installed.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

include(
    ":koblas",
    ":koblas-bench",
    ":koblas-bench:jmh",
)
