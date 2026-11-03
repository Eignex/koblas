@file:kotlin.jvm.JvmName("WorkspaceEngines")

package com.eignex.koblas.workspace

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.arithmeticEngine

/**
 * Uses [workspace] for matrix alias snapshots, packed panels and sparse scratch.
 *
 * Arithmetic and routing stay those of this engine. Repeated bindings reuse the same adapter; binding an
 * already bound engine replaces its workspace. A null workspace returns the underlying arithmetic engine.
 * The workspace supports nested calls but must not be shared by concurrent calls.
 */
public fun KoblasEngine.withWorkspace(workspace: Workspace?): KoblasEngine = workspace?.bind(this)
    ?: arithmeticEngine
