plugins {
    `kotlin-dsl`
    `java-gradle-plugin`
}

group = "com.eignex.koblas"

gradlePlugin {
    plugins {
        create("koblasNativeLibrary") {
            id = "koblas.native-library"
            implementationClass = "com.eignex.koblas.buildlogic.KoblasNativeLibraryPlugin"
        }
        create("koblasBuildScriptStructure") {
            id = "koblas.build-structure"
            implementationClass = "com.eignex.koblas.buildlogic.KoblasBuildScriptStructurePlugin"
        }
    }
}
