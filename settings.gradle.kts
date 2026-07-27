rootProject.name = "koblas"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

include(":koblas", ":koblas-openblas", ":koblas-bench")
