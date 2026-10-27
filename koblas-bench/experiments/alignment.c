#include <stdint.h>

/* Diagnostic snapshot outside arithmetic; no production kernel calls this function. */
uint64_t array_page_offset(const void *array) {
    return (uintptr_t)array & 4095;
}
