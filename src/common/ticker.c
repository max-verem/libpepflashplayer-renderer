#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#include "ticker.h"

int64_t ticker_now()
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return tm.tv_nsec + tm.tv_sec * 1000000000LL;
};

int ticker_init(ticker_t** ptick, int64_t period_ns)
{
    ticker_t* tick;

    if(!ptick || !period_ns)
        return -EINVAL;

    *ptick = tick = (ticker_t*)calloc(1, sizeof(ticker_t));

    if(!tick)
        return -ENOMEM;

    tick->period_ns = period_ns;
    tick->start = ticker_now();

    return 0;
};

int ticker_release(ticker_t** ptick)
{
    ticker_t* tick;

    if(!ptick || !*ptick)
        return -EINVAL;

    tick = *ptick;
    *ptick = NULL;

    free(tick);

    return 0;
};

int ticker_wait(ticker_t* tick)
{
    struct timespec tm;
    int64_t now, next;

    next = ticker_now();
    next += tick->period_ns - (next % tick->period_ns);

    tm.tv_sec = next / 1000000000LL;
    tm.tv_nsec = next % 1000000000LL;

    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tm, NULL);

    now = ticker_now();

    return now - next;
};
