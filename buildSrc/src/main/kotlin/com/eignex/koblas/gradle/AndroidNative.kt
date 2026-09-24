package com.eignex.koblas.gradle

import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.ArchiveOperations
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemOperations
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.CacheableTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import java.net.URI
import java.security.MessageDigest
import javax.inject.Inject

/** Downloads one file and refuses it unless it has the expected SHA-256. */
abstract class DownloadVerified : DefaultTask() {
    @get:Input abstract val url: Property<String>

    @get:Input abstract val sha256: Property<String>

    @get:OutputFile abstract val destination: RegularFileProperty

    @TaskAction
    fun download() {
        val target = destination.get().asFile
        target.parentFile.mkdirs()
        URI(url.get()).toURL().openStream().use { input -> target.outputStream().use { input.copyTo(it) } }
        val digest = MessageDigest.getInstance("SHA-256").digest(target.readBytes())
            .joinToString("") { "%02x".format(it) }
        if (digest != sha256.get()) {
            target.delete()
            throw GradleException("${url.get()} has SHA-256 $digest, expected ${sha256.get()}")
        }
    }
}

/**
 * Cross-compiles one static OpenBLAS for one ABI: CBLAS only, no threads, and locking kept so concurrent
 * callers stay safe on a library built without a thread pool.
 *
 * `USE_SIMPLE_THREADED_LEVEL3` changes nothing in a build without threads. It is set because without it the
 * level 3 driver compiles its threaded syrk objects anyway, and those only compile in a threaded build.
 */
@CacheableTask
abstract class BuildAndroidOpenBlas @Inject constructor(
    private val exec: ExecOperations,
    private val files: FileSystemOperations,
    private val archives: ArchiveOperations,
) : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val source: RegularFileProperty

    @get:Input abstract val version: Property<String>

    @get:Input abstract val ndkVersion: Property<String>

    @get:Input abstract val minSdk: Property<Int>

    @get:Input abstract val triple: Property<String>

    @get:Input abstract val target: Property<String>

    @get:Input abstract val cores: Property<String>

    @get:Internal abstract val toolchain: DirectoryProperty

    @get:Internal abstract val workDirectory: DirectoryProperty

    @get:OutputFile abstract val library: RegularFileProperty

    @TaskAction
    fun build() {
        val bin = toolchain.get().asFile
        if (!bin.isDirectory) throw GradleException("NDK ${ndkVersion.get()} is not installed: $bin")
        val work = workDirectory.get().asFile
        files.delete { delete(work) }
        files.copy {
            from(archives.tarTree(archives.gzip(source)))
            into(work)
        }
        val tree = work.resolve("OpenBLAS-${version.get()}")
        exec.exec {
            workingDir(tree)
            commandLine(
                "make", "-j${Runtime.getRuntime().availableProcessors()}",
                "CC=${bin.resolve("${triple.get()}${minSdk.get()}-clang")}",
                "AR=${bin.resolve("llvm-ar")}", "RANLIB=${bin.resolve("llvm-ranlib")}", "HOSTCC=cc",
                "NOFORTRAN=1", "ONLY_CBLAS=1", "C_LAPACK=0", "NO_LAPACK=1", "NO_LAPACKE=1",
                "TARGET=${target.get()}", "DYNAMIC_ARCH=1", "DYNAMIC_LIST=${cores.get()}",
                "USE_THREAD=0", "USE_LOCKING=1", "NUM_THREADS=1", "USE_SIMPLE_THREADED_LEVEL3=1",
                "NO_SHARED=1", "BINARY=64", "libs",
            )
        }
        tree.resolve("libopenblas-r${version.get()}.a").copyTo(library.get().asFile, overwrite = true)
        files.delete { delete(work) }
    }
}

/**
 * Links the JNI shim against one ABI's OpenBLAS into `<abi>/libkoblas_openblas.so`, the layout `jniLibs` takes.
 *
 * `--exclude-libs` hides every OpenBLAS symbol, so the library exports its JNI entry points and nothing an
 * application's own BLAS could collide with. The 16 KB page alignment is what Android 15 and later require.
 */
@CacheableTask
abstract class LinkAndroidShim @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.RELATIVE) abstract val shim: RegularFileProperty

    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val openBlas: RegularFileProperty

    @get:Input abstract val abi: Property<String>

    @get:Input abstract val ndkVersion: Property<String>

    @get:Input abstract val minSdk: Property<Int>

    @get:Input abstract val triple: Property<String>

    @get:Internal abstract val toolchain: DirectoryProperty

    @get:OutputDirectory abstract val outputDirectory: DirectoryProperty

    @TaskAction
    fun link() {
        val bin = toolchain.get().asFile
        val out = outputDirectory.get().asFile.resolve(abi.get()).apply { mkdirs() }
        val unstripped = temporaryDir.resolve("libkoblas_openblas.so")
        exec.exec {
            commandLine(
                bin.resolve("${triple.get()}${minSdk.get()}-clang"), "-O2", "-fPIC", "-shared",
                "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror",
                "-fvisibility=hidden", "-ffunction-sections", "-fdata-sections",
                "-o", unstripped, shim.get().asFile, openBlas.get().asFile, "-lm",
                "-Wl,--gc-sections", "-Wl,--exclude-libs,ALL", "-Wl,-z,max-page-size=16384", "-Wl,--no-undefined",
            )
        }
        exec.exec {
            commandLine(bin.resolve("llvm-strip"), "--strip-unneeded", "-o", out.resolve(unstripped.name), unstripped)
        }
    }
}

/**
 * Rewrites a test APK's merged manifest to declare it not debuggable.
 *
 * AGP marks every test APK debuggable and overrides a manifest that says otherwise. A debuggable process runs
 * ART in its debuggable mode, where a JNI transition and much of the JIT's work cost what they never cost in a
 * release application, so a crossover measured in one would be placed for a build nobody ships.
 */
@CacheableTask
abstract class ReleaseLikeTestManifest : DefaultTask() {
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) abstract val merged: RegularFileProperty

    @get:OutputFile abstract val updated: RegularFileProperty

    @TaskAction
    fun rewrite() {
        val text = merged.get().asFile.readText()
        val flag = "android:debuggable=\"true\""
        if (flag !in text) throw GradleException("the merged test manifest does not declare $flag")
        updated.get().asFile.writeText(text.replace(flag, "android:debuggable=\"false\""))
    }
}
