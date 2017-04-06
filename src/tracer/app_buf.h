#ifndef APP_BUF_H
#define APP_BUF_H

#include <pthread.h>

#define APP_BUFFERS_MAX 8
typedef struct app_buffer_desc
{
    int cnt, pinned, size;
    pthread_mutex_t lock;
    int ref[APP_BUFFERS_MAX];
    void* buf[APP_BUFFERS_MAX];
} app_buffer_t;

int app_buffer_init(app_buffer_t* ab, int cnt, int size, int pinned);
int app_buffer_release(app_buffer_t* ab);
int app_buffer_unref(app_buffer_t* ab, int idx);
int app_buffer_ref(app_buffer_t* ab, int idx);
int app_buffer_idx(app_buffer_t* ab, void* ptr);

#endif /* APP_BUF_H */
