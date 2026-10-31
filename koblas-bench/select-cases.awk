# Filter before smoke truncation; leave tags for the runners to validate and omit from case IDs.
BEGIN {
    FS = "+"
    if (suite != "default" && suite != "sweep") fail("suite must be default or sweep")
    if (suite == "sweep" && operation == "all") fail("suite sweep requires a specific operation")
}
function fail(message) {
    print "error: " message > "/dev/stderr"
    failed = 1
    exit 2
}
{
    sub(/^[ \t]+/, "")
    sub(/[ \t\r]+$/, "")
    if ($0 == "" || /^#/) next
    membership = "default"
    seen = 0
    for (i = 4; i <= NF; i++) {
        if ($i ~ /^suite=/) {
            if (seen++) fail("duplicate suite at line " NR)
            membership = substr($i, 7)
            if (membership != "default" && membership != "sweep" &&
                membership != "default,sweep" && membership != "sweep,default") {
                fail("invalid suite at line " NR)
            }
        }
    }
    if (index("," membership ",", "," suite ",") && (operation == "all" || $1 == operation)) {
        if (smoke != "true" || count < 3) print
        count++
    }
}
END {
    if (!failed && !count) fail("suite and operation selected no cases")
}
