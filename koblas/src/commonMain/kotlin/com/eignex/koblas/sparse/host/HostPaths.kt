package com.eignex.koblas.sparse.host

/**
 * The directory part of [path], or null when it names no directory to look in.
 *
 * A bundled artifact extracts a whole library collection into one directory, so the sibling a binding wants
 * is the one beside the library it already loaded. Each target spells the split its own way, which is all
 * this is for.
 */
internal expect fun parentDirectory(path: String): String?

/**
 * Where to look for a host library, in the order a loader should try them: the [explicit] path a caller
 * configured, then the siblings in [searchDirectory], then the platform [sonames] as the loader resolves
 * them itself.
 *
 * A bundled artifact extracts a whole library collection into one directory, so the sibling search is what
 * lets a binding find the copy beside the library it already loaded. A soname that already names a
 * directory is left for the loader, since prefixing it would name nothing.
 */
internal fun hostLibraryCandidates(sonames: List<String>, explicit: String?, searchDirectory: String?): List<String> =
    buildList {
        explicit?.let(::add)
        searchDirectory?.let { directory ->
            for (soname in sonames) if ('/' !in soname) add("$directory/$soname")
        }
        addAll(sonames)
    }
