#ifndef APP_H
#define APP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "drvapi_error_string.h"

#include <Processing.NDI.Lib.h>

#include "instance.h"
#include "app_buf.h"

#define APP_OPS_SIZE    10
#define APP_STRM_SIZE   5

typedef struct app_desc
{
    app_buffer_t buf_host, buf_dev;
    int f_exit, swaps, vga[2], ops[APP_OPS_SIZE], ndi_sends;
    pthread_mutex_t lock;
    size_t size, stride;
    instance_t *inst;
    CUcontext cu_ctx;
    cudaStream_t cu_streams[APP_STRM_SIZE];
    int cu_dev;
    NDIlib_send_instance_t m_p_ndi_send;
} app_t;

int app_create(app_t** p_app, instance_t* inst);
int app_destroy(app_t** p_app);
int app_run(app_t* app);
int app_stop(app_t* app);

#endif
