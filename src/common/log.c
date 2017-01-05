#include <stdio.h>
#include <stdarg.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

#include "log.h"

pthread_mutex_t _log_lock = PTHREAD_MUTEX_INITIALIZER;

static int _log_level = 20;

void log_level(int level)
{
    _log_level = level;
};

static void log_message_v(int level, const char* file, const int line, const char* function, const char *fmt, va_list ap)
{
    char buf[64];
    struct timeval tv;
    struct tm result;

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &result);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &result);

    pthread_mutex_lock(&_log_lock);
    fprintf(stderr, "%s.%.3d [%s:%d:%s] ", buf, (int)(tv.tv_usec / 1000), file, line, function);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    pthread_mutex_unlock(&_log_lock);
};

void log_message(int level, const char* file, const int line, const char* function, const char *fmt, ...)
{
    va_list ap;

    if(level > _log_level)
        return;

    va_start(ap, fmt);

    log_message_v(level, file, line, function, fmt, ap);

    va_end(ap);
};


