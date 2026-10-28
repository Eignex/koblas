# Android instrumentation can exit zero after an assertion failure, crash, or empty run.
function reject(message) {
    print message > "/dev/stderr"
    invalid = 1
    exit 1
}
function xml(value) {
    gsub(/&/, "\\&amp;", value)
    gsub(/</, "\\&lt;", value)
    gsub(/>/, "\\&gt;", value)
    gsub(/"/, "\\&quot;", value)
    return value
}
{ sub(/\r$/, "") }
/^INSTRUMENTATION_FAILED:/ { reject($0) }
/^INSTRUMENTATION_STATUS: / {
    field = substr($0, length("INSTRUMENTATION_STATUS: ") + 1)
    separator = index(field, "=")
    if (separator) fields[substr(field, 1, separator - 1)] = substr(field, separator + 1)
}
/^INSTRUMENTATION_STATUS_CODE: / {
    if ($2 !~ /^-?[0-9]+$/) reject("invalid instrumentation status")
    code = $2 + 0
    if (!fields["class"] || !fields["test"]) reject("instrumentation status has no test identity")
    identity = fields["class"] SUBSEP fields["test"]
    if (code == 1) {
        if (identity in pending) reject("instrumentation started a test twice")
        pending[identity] = 1
    } else {
        if (code != 0 && code != -1 && code != -2 && code != -3 && code != -4)
            reject("unexpected instrumentation status " code)
        if (!(identity in pending)) reject("instrumentation finished a test without starting it")
        delete pending[identity]
        tests++
        cases = cases "<testcase classname=\"" xml(fields["class"]) "\" name=\"" xml(fields["test"]) "\" time=\"0\">"
        kind = ""
        if (code == -1) { errors++; kind = "error" }
        if (code == -2) { failures++; kind = "failure" }
        if (code == -3 || code == -4) { skipped++; kind = "skipped" }
        if (kind) cases = cases "<" kind ">" xml(fields["stack"] fields["stream"]) "</" kind ">"
        cases = cases "</testcase>\n"
    }
    for (key in fields) delete fields[key]
}
/^INSTRUMENTATION_CODE: / { final_code = $2; finished = 1 }
/^Time: [0-9.]+$/ { elapsed = $2 }
END {
    if (invalid) exit 1
    if (!finished || final_code != -1) reject("instrumentation did not finish successfully")
    for (key in pending) reject("instrumentation did not finish successfully")
    if (!tests) reject("instrumentation executed no tests")
    print "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    printf "<testsuite name=\"%s\" tests=\"%d\" failures=\"%d\" errors=\"%d\" skipped=\"%d\" time=\"%.3f\">\n", \
        xml(suite), tests, failures, errors, skipped, elapsed
    printf "%s</testsuite>\n", cases
    exit (failures || errors || skipped == tests)
}
