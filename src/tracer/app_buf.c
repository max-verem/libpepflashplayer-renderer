#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "drvapi_error_string.h"

#include "log.h"
#include "app_buf.h"

int app_buffer_init(app_buffer_t* ab, int cnt, int size, int pinned)
{
    int i;

    ab->cnt = cnt;
    ab->size = size;
    ab->pinned = pinned;

    LOG_N("cnt=%d, size=%d, pinned=%d", cnt, size, pinned);

    pthread_mutex_init(&ab->lock, NULL);

    for(i = 0; i < cnt && i < APP_BUFFERS_MAX; i++)
    {
        ab->ref[i] = 0;

        if(pinned)
            cudaMallocHost(&ab->buf[i], size);
        else
            cudaMalloc(&ab->buf[i], size);
    };

    return 0;
};

int app_buffer_release(app_buffer_t* ab)
{
    int i;

    if(!ab)
        return -EINVAL;

    pthread_mutex_destroy(&ab->lock);

    for(i = 0; i < ab->cnt && i < APP_BUFFERS_MAX; i++)
    {
        ab->ref[i] = 0;

        if(ab->pinned)
            cudaFreeHost(ab->buf[i]);
        else
            cudaFree(ab->buf[i]);
    };

    return 0;
};

int app_buffer_unref(app_buffer_t* ab, int idx)
{
    int r = -ENOENT;

    if(!ab)
        return -EINVAL;

    pthread_mutex_lock(&ab->lock);

    if(idx >= 0 && idx < APP_BUFFERS_MAX && ab->ref[idx])
        r = ab->ref[idx]--;

    pthread_mutex_unlock(&ab->lock);

    return r;
};

int app_buffer_ref(app_buffer_t* ab, int idx)
{
    int i, r = -ENOENT;

    if(!ab)
        return -EINVAL;

    pthread_mutex_lock(&ab->lock);

    if(idx >= 0)
    {
        if(idx >= APP_BUFFERS_MAX)
            r = -ENOENT;
        else
            r = ab->ref[idx]++;
    }
    else
    {
        for(i = 0; i < ab->cnt && i < APP_BUFFERS_MAX && r < 0; i++)
            if(!ab->ref[i])
                ab->ref[r = i]++;
    };

    pthread_mutex_unlock(&ab->lock);

    return r;
};

int app_buffer_idx(app_buffer_t* ab, void* ptr)
{
    int i;

    if(!ptr || !ab)
        return -EINVAL;

    for(i = 0; i < ab->cnt && i < APP_BUFFERS_MAX; i++)
        if(ptr == ab->buf[i])
            return i;

    return -ENOENT;
};
