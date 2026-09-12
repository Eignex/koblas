@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.internal.kernels

import kotlinx.cinterop.*

/** Uses the target archive's C discovery, with no independent Kotlin CPU inference. */
internal actual object NativeProbe {
    actual fun query(kind: Int, index: Int, kernelId: Int): IntArray? = memScoped {
        val request = alloc<koblas_probe_request_v1>()
        val result = alloc<koblas_probe_result_v1>()
        request.ptr.reinterpret<UIntVar>().let { words -> for (i in 0 until 8) words[i] = 0u }
        request.abi_version = 1u
        request.byte_size = 32u
        request.query = kind.toUInt()
        request.index = index.toUInt()
        request.kernel_id = kernelId.toUInt()
        result.abi_version = 1u
        result.byte_size = 256u
        val status = koblas_probe_v1(request.ptr, result.ptr)
        check(status == 0) { "native probe rejected: $status" }
        val words = result.ptr.reinterpret<UIntVar>()
        IntArray(64) { words[it].toInt() }
    }
}
