#ifndef TICKER_H
#define TICKER_H

#include <stdint.h>

typedef struct ticker_desc
{
    int64_t period_ns, start;
} ticker_t;

int ticker_init(ticker_t** ptick, int64_t period_ns);
int ticker_release(ticker_t** ptick);
int ticker_wait(ticker_t* tick);
int64_t ticker_now();

#endif /* TICKER_H */
