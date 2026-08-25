/*
 * Stands in for HiGHS's io/HighsIO.cpp. HFactor reaches the rest of HiGHS only to log, while that file
 * pulls in the options parser and the external-API loader, so it is replaced rather than linked.
 */
#include "io/HighsIO.h"

#include <cstdarg>
#include <cstdio>

static void emit(const HighsLogOptions& options, const char* format, va_list args) {
    if (options.log_stream == nullptr) return;
    vfprintf(options.log_stream, format, args);
}

void highsLogDev(const HighsLogOptions& log_options_, const HighsLogType, const char* format, ...) {
    if (log_options_.log_dev_level == nullptr || *log_options_.log_dev_level == 0) return;
    va_list args;
    va_start(args, format);
    emit(log_options_, format, args);
    va_end(args);
}

void highsLogUser(const HighsLogOptions& log_options_, const HighsLogType, const char* format, ...) {
    va_list args;
    va_start(args, format);
    emit(log_options_, format, args);
    va_end(args);
}
