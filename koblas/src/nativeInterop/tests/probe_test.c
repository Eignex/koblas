#define _GNU_SOURCE
#include "koblas_kernels.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static koblas_probe_request_v1 request(uint32_t kind) {
    koblas_probe_request_v1 value = {1, sizeof(value), kind, 0, 0, {0}};
    return value;
}
static koblas_probe_result_v1 result(void) {
    koblas_probe_result_v1 value = {0};
    value.abi_version = 1; value.byte_size = sizeof(value);
    return value;
}
/* A trailing protected page makes any unpredicated tail overread or overwrite observable. */
static void guarded_vectors(uint32_t variant) {
    const int lengths[] = {1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1025};
    size_t page = (size_t)sysconf(_SC_PAGESIZE), readable = page * 4;
    void *memory = mmap(NULL, readable + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(memory != MAP_FAILED && mprotect((char *)memory + readable, page, PROT_NONE) == 0);
    for (size_t j = 0; j < sizeof(lengths) / sizeof(lengths[0]); j++) {
        int len = lengths[j];
        double *values = (double *)((char *)memory + readable) - len;
        double expected = 0.0, actual = -1.0;
        for (int i = 0; i < len; i++) { values[i] = i * 0.125; expected += values[i] * values[i]; }
        assert(koblas_dense_dot_v1(KOBLAS_OP_DENSE_DOT * 16 + variant, values, 0, values, 0, len, &actual) == KOBLAS_OK);
        assert(actual == expected);
        assert(koblas_dense_scale_v1(KOBLAS_OP_DENSE_SCALE * 16 + variant, values, 0, 1.25, len) == KOBLAS_OK);
        for (int i = 0; i < len; i++) assert(values[i] == (i * 0.125) * 1.25);
    }
    assert(munmap(memory, readable + page) == 0);
}

int main(void) {
    koblas_probe_request_v1 q = request(KOBLAS_HOST);
    koblas_probe_result_v1 r = result();
    assert(koblas_probe_v1(NULL, &r) == KOBLAS_INVALID_ARGUMENT);
    assert(koblas_probe_v1(&q, NULL) == KOBLAS_INVALID_ARGUMENT);
    r.byte_size = 8;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_BUFFER_TOO_SMALL && r.byte_size == 8);
    r = result(); r.abi_version = 2;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_BAD_VERSION && r.abi_version == 2);
    r = result(); q.byte_size = 8;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_BUFFER_TOO_SMALL);
    q = request(KOBLAS_HOST); q.abi_version = 2;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_BAD_VERSION);
    q = request(KOBLAS_HOST); q.reserved[0] = 1;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_INVALID_ARGUMENT);
    q = request(100);
    assert(koblas_probe_v1(&q, &r) == KOBLAS_INVALID_ARGUMENT);
    struct { koblas_probe_request_v1 value; uint32_t extra; } extended_q = {request(KOBLAS_HOST), 0};
    struct { koblas_probe_result_v1 value; uint32_t guard; } extended_r = {result(), 0xf00dcafe};
    extended_q.value.byte_size = sizeof(extended_q);
    extended_r.value.byte_size = sizeof(extended_r);
    assert(koblas_probe_v1(&extended_q.value, &extended_r.value) == KOBLAS_OK);
    assert(extended_r.guard == 0xf00dcafe && extended_r.value.byte_size == sizeof(r));
    extended_q.extra = 1;
    assert(koblas_probe_v1(&extended_q.value, &r) == KOBLAS_INVALID_ARGUMENT);
    q = request(KOBLAS_HOST);
    assert(koblas_probe_v1(&q, &r) == KOBLAS_OK);
    unsigned count = r.kernel_count;
    printf("host arch=%u os=%u hardware=%x usable=%x built=%x kernels=%u\n",
        r.architecture, r.operating_system, r.hardware_features, r.os_features, r.built_features, count);
    for (unsigned i = 0; i < count; i++) {
        q = request(KOBLAS_KERNEL); q.index = i;
        assert(koblas_probe_v1(&q, &r) == KOBLAS_OK);
        unsigned id = r.kernel_id;
        if (r.operation == KOBLAS_OP_DENSE_DOT && r.reason == KOBLAS_OK) guarded_vectors(r.variant);
        if (r.operation == KOBLAS_OP_DENSE_DOT && r.reason != KOBLAS_OK) {
            double untouched = 91;
            assert(koblas_dense_dot_v1(id, NULL, 0, NULL, 0, 1, &untouched) == (int32_t)r.reason);
            assert(untouched == 91);
        }
        assert(id / 16u == r.operation && id % 16u == r.variant);
        assert(r.process_state == 0 && r.thread_state == 0 && r.call_state == 0);
        assert(r.a_type == KOBLAS_F64 && r.accumulation_type == KOBLAS_F64);
        koblas_probe_result_v1 by_id = result();
        q.index = 0; q.kernel_id = id;
        assert(koblas_probe_v1(&q, &by_id) == KOBLAS_OK);
        assert(memcmp(&r, &by_id, sizeof(r)) == 0);
        printf("kernel=%u operation=%u variant=%u reason=%u bits=%u batch=%u unroll=%u\n",
            id, r.operation, r.variant, r.reason, r.register_bits, r.logical_batch, r.unroll);
    }
    q = request(KOBLAS_KERNEL); q.index = count;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_NOT_FOUND);
    q = request(KOBLAS_KERNEL); q.kernel_id = 0xffffffff;
    assert(koblas_probe_v1(&q, &r) == KOBLAS_NOT_FOUND);
    q = request(KOBLAS_THREAD);
    assert(koblas_probe_v1(&q, &r) == KOBLAS_OK);
    assert(r.ordinary_vl_bytes == 0 || (r.context_validity & 1));
    assert(r.streaming_vl_bytes == 0 || (r.context_validity & 2));
    double value = 123;
    assert(koblas_dense_scale_v1(0xffffffff, &value, 0, 2, 1) == KOBLAS_NOT_FOUND && value == 123);
    assert(koblas_dense_scale_v1(KOBLAS_OP_DENSE_DOT * 16 + KOBLAS_SCALAR, &value, 0, 2, 1) == KOBLAS_NOT_FOUND && value == 123);
    assert(koblas_dense_dot_v1(KOBLAS_OP_DENSE_DOT * 16 + KOBLAS_SCALAR, NULL, 0, NULL, 0, 0, &value) == KOBLAS_OK && value == 0);
    assert(koblas_dense_axpy_v1(KOBLAS_OP_DENSE_AXPY * 16 + KOBLAS_SCALAR, NULL, 0, 0, NULL, 0, 100) == KOBLAS_OK);
    puts("probe ABI and rejection checks passed");
}
