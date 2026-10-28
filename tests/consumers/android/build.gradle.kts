plugins { id("com.android.application") }

val consumerVersion = providers.gradleProperty("consumer.version").orElse("0.0.0-consumer")

android {
    namespace = "com.eignex.koblas.consumer"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.eignex.koblas.consumer"
        minSdk = 24
        targetSdk = 36
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    sourceSets.getByName("main").kotlin.directories.add("../shared")
    testBuildType = providers.gradleProperty("consumer.buildType").orElse("debug").get()
    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "consumer-rules.pro")
            testProguardFiles("test-rules.pro")
        }
    }
}

dependencies {
    implementation("com.eignex:koblas:${consumerVersion.get()}")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    // R8 also shrinks the test APK, whose runner references these compile-only annotations.
    androidTestImplementation("com.google.errorprone:error_prone_annotations:2.50.0")
}
