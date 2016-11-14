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
};

typedef struct url_loader_desc
{
    PP_Instance instance_id;
    PP_Resource self;

    PP_Resource request_info;
    url_request_info_t* url_request_info;

    pthread_t th;
    int f_curl, state;

    FILE* reader;

    int64_t bytes_received;
    int64_t total_bytes_to_be_received;

    struct
    {
        struct PP_Var STATUSCODE;
        struct PP_Var HEADERS;
    } response;

} url_loader_t;

#endif /* PPB_URLLoader_h */
