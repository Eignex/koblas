#define main vendor_runner_main
#include "vendor_runner.c"
#undef main
#include <sys/wait.h>
#include <unistd.h>

static void require_test(int condition, const char *message) { if (!condition) fail(message); }

static bench_case parse_test_case(const char *text) {
    char line[2048];
    copy_field(line, sizeof(line), text, 1, "test case");
    bench_case result = {0};
    require_test(parse_case(line, 1, &result), "expected a case");
    return result;
}

static void expect_failure(const char *text, const char *suite, const char *operation) {
    pid_t child = fork();
    if (child < 0) fail("fork failed");
    if (!child) {
        if (!freopen("/dev/null", "w", stderr)) _exit(3);
        bench_case spec = parse_test_case(text);
        if (suite) select_cases(&spec, 1, suite, operation);
        _exit(0);
    }
    int status = 0;
    require_test(waitpid(child, &status, 0) == child, "waitpid failed");
    require_test(WIFEXITED(status) && WEXITSTATUS(status) == 2, "expected selection or parsing failure");
}

int main(void) {
    const char *ids[] = {"dot+4096+uniform", "gemm-block+15x7x31+uniform+packed=4x4+timing=prepacked-compute"};
    const char *memberships[] = {"default", "sweep", "default,sweep", "sweep,default"};
    for (size_t i = 0; i < sizeof(ids) / sizeof(*ids); ++i) {
        bench_case original = parse_test_case(ids[i]);
        for (size_t j = 0; j < sizeof(memberships) / sizeof(*memberships); ++j) {
            char line[2048];
            snprintf(line, sizeof(line), "%s+suite=%s", ids[i], memberships[j]);
            bench_case tagged = parse_test_case(line);
            require_test(!strcmp(original.id, tagged.id), "suite changed case identity");
            require_test(original.option_count == tagged.option_count, "suite became a workload option");
        }
    }
    bench_case cases[] = {
        parse_test_case("dot+4096+uniform+suite=default,sweep"),
        parse_test_case("sum+4096+uniform"),
        parse_test_case("dot+7+uniform+suite=sweep"),
        parse_test_case("sum+7+uniform+suite=sweep"),
    };
    bench_case selected[4];
    memcpy(selected, cases, sizeof(cases));
    require_test(select_cases(selected, 4, "default", "all") == 2, "default included sweep cases");
    memcpy(selected, cases, sizeof(cases));
    require_test(select_cases(selected, 4, "default", "dot") == 1, "operation did not intersect default");
    memcpy(selected, cases, sizeof(cases));
    require_test(select_cases(selected, 4, "sweep", "dot") == 2, "operation did not intersect sweep");
    require_test(!strcmp(selected[0].id, "dot+4096+uniform") && !strcmp(selected[1].id, "dot+7+uniform"),
        "shared case duplicated or selection order changed");
    const char *invalid[] = {"", "all", "default,default", "sweep,", ",sweep", "default,other", "default+suite=sweep"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
        char line[256];
        snprintf(line, sizeof(line), "dot+7+uniform+suite=%s", invalid[i]);
        expect_failure(line, NULL, NULL);
    }
    expect_failure("dot+7+uniform+suite=sweep", "sweep", "all");
    expect_failure("dot+7+uniform+suite=sweep", "other", "dot");
    expect_failure("dot+7+uniform+suite=sweep", "default", "dot");
    expect_failure("dot+7+uniform+suite=sweep", "sweep", "sum");
    puts("vendor case selection checks passed");
    return 0;
}
