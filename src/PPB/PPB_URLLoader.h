#ifndef PPB_URLLoader_h
#define PPB_URLLoader_h

#include <pthread.h>
#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_resource.h>
#include <ppapi/c/pp_var.h>

#include "PPB_URLRequestInfo.h"

enum url_loader_states
{
    URLLoader_NONE = 0,
    URLLoader_CONNECTING,
    URLLoader_DOWNLOADING,
    URLLoader_DONE,
    URLLoader_ABORTED,
    URLLoader_ERROR,
};

typedef struct url_loader_desc
{
    PP_Instance instance_id;
    PP_Resource self;
    PP_Resource request_info;

    enum url_loader_states state;

    void* curl;
    pthread_t thread;
    pthread_mutex_t lock;
    int f_thread_started, f_thread_abort;

    struct
    {
        char* header_buffer;
        int64_t header_len;
        int64_t body_len;
        int64_t body_size;
        int64_t body_pos;
        int statuscode;
    } recv;

    struct
    {
        int64_t body_pos;
    } sent;

    struct PP_CompletionCallback callback_connect;
    struct PP_CompletionCallback callback_data;

    FILE* reader;
    FILE* writer;

} url_loader_t;

#endif /* PPB_URLLoader_h */
