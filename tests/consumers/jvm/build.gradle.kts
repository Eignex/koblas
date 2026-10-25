import org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget

plugins { kotlin("multiplatform") }

kotlin {
    jvmToolchain(17)
    jvm()
    sourceSets {
        commonMain {
            kotlin.srcDir("../shared")
            dependencies { implementation("com.eignex:koblas:0.0.0-consumer") }
        }
    }
}

val main = (kotlin.targets.getByName("jvm") as KotlinJvmTarget).compilations.getByName("main")
for (jdk in listOf(17, 25)) {
    tasks.register<JavaExec>("jvm${jdk}Consumer") {
        dependsOn(main.compileTaskProvider)
        classpath(main.output.allOutputs, main.runtimeDependencyFiles)
        mainClass.set("com.eignex.koblas.consumer.MainKt")
        javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(jdk)) })
        if (jdk == 25) {
            jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
            args("simd")
        }
    }
}

for (jdk in listOf(17, 25)) {
    tasks.register<JavaExec>("jvm${jdk}ModuleConsumer") {
        dependsOn(main.compileTaskProvider)
        classpath(main.output.allOutputs)
        mainClass.set("com.eignex.koblas.consumer.MainKt")
        javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(jdk)) })
        jvmArgs("--module-path", main.runtimeDependencyFiles.asPath)
        if (jdk == 25) {
            jvmArgs("--add-modules=com.eignex.koblas,jdk.incubator.vector", "--enable-native-access=com.eignex.koblas")
            args("simd")
        } else {
            jvmArgs("--add-modules=com.eignex.koblas")
        }
    }
}
