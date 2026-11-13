plugins { java }

repositories { mavenCentral() }

java { toolchain.languageVersion.set(JavaLanguageVersion.of(25)) }

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:6.1.3")
    testImplementation("org.jetbrains.kotlin:kotlin-stdlib:2.4.21")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:6.1.3")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs("--add-modules=jdk.incubator.vector")
}

tasks.compileTestJava { options.compilerArgs.add("--add-modules=jdk.incubator.vector") }
