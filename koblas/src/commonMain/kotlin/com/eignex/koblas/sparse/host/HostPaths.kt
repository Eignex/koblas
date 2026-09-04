package com.eignex.koblas.sparse.host

/**
 * The directory part of [path], or null when it names no directory to look in.
 *
 * A bundled artifact extracts a whole library collection into one directory, so the sibling a binding wants
 * is the one beside the library it already loaded. Each target spells the split its own way, which is all
 * this is for.
 */
internal expect fun parentDirectory(path: String): String?
