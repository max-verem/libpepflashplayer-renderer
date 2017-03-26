#include <dlfcn.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "drvapi_error_string.h"

#include "log.h"
#include "mod.h"
#include "res.h"
#include "instance.h"
#include "if.h"

#include "PPB_MessageLoop.h"
#include "PPB_Var.h"

#define SWF_ARGS "line1=foo_url1&line2=bar_url2&line3=foo_url3&line4=bar_url4&param1=DEMO1%20LONG%20String"
#define SWF_PATH "file:///usr/local/src/libpepflashplayer-renderer.git/tests/src"

//#define SWF_NAME "m1-lowerThird-1080i50.swf"
#define SWF_NAME "as3_test4.swf"

//#define SWF_NAME "1080i50-blank_test_GPU.swf"
//#define SWF_NAME "m1_logo_1080i50_BIG.swf"
//#define SWF_NAME "m1_logo_1080i50.swf"
//#define SWF_NAME "demo4.swf"
//#define SWF_NAME "demo1_2_movie.swf"
//#define SWF_NAME "demo1_2_image.swf"
//#define SWF_NAME "demo2.swf"

const char* so_name = "/usr/local/src/libpepflashplayer-renderer.git/tests/libpepflashplayer.so-24.0.0.186-debug";
const char* local_path = "/usr/local/src/libpepflashplayer-renderer.git/tests/local";

const char* argn[] = { /* "src", */ "quality", /* "bgcolor", "width", "height",*/ "wmode", "AllowScriptAccess", /* "flashvars", */ NULL};
const char* argv[] = { /* SWF, */ "high", /* "#ece9d8",  "1920", "1080", */"transparent", "Always", /* "line1=foo_flashvars&line2=bar_flashvars", */ NULL};

static void f1(void* user_data, int32_t result)
{
    LOG_TD;
};

static mod_t* mod = NULL;

instance_t* inst = NULL;


static void sighandler(int sig)
{
    int r = -1;

    LOG_N("sig=%d", sig);

    if(mod && inst)
        r = mod->interface.message_loop->PostQuit(inst->message_loop_id, 0);

    LOG_N("r=%d", r);
};

static size_t shmSize = 1920 * 1080 * 4;
static void* shmPtr = NULL;
static cudaIpcMemHandle_t shmHandle;
static CUcontext shmCU_CTX;
static int shmCU_DEV = 0;

static int pop_cuda_shmem_handle(struct instance_desc* inst, void* phandle, size_t* sz)
{
    LOG_N("inst=%p", inst);

    *((cudaIpcMemHandle_t*)phandle) = shmHandle;
    *sz = shmSize;

    return 0;
};

static int push_cuda_shmem_handle(struct instance_desc* inst, void* phandle)
{
    LOG_N("inst=%p, phandle=%p", inst, phandle);
    return 0;
};

int main()
{
    int r, instance_id;

    {
        CUresult e;
        CUcontext cu_ctx_pop;

        if(CUDA_SUCCESS != (e = cuInit(0)))
        {
            LOG_E("cuInit failed");
            return 0;
        };

        if(CUDA_SUCCESS != (e = cuCtxCreate(&shmCU_CTX, CU_CTX_BLOCKING_SYNC, shmCU_DEV)))
        {
            LOG_E("cuCtxCreate failed: %s", getCudaDrvErrorString(e));
            return 0;
        };

        // init cuda data here
        if(CUDA_SUCCESS != (e = cudaMalloc(&shmPtr, shmSize)))
        {
            LOG_E("cudaMalloc failed: %s", getCudaDrvErrorString(e));
            return 0;
        };

        if(CUDA_SUCCESS != (e = cudaIpcGetMemHandle(&shmHandle, shmPtr)))
        {
            LOG_E("cudaIpcGetMemHandle failed: %s", getCudaDrvErrorString(e));
            return 0;
        };

        if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
        {
            LOG_E("cuCtxPopCurrent failed: %s", getCudaDrvErrorString(e));
            return 0;
        };
    };

//    log_level(100);

    res_begin();

    instance_id = res_create(sizeof(instance_t), NULL, NULL);
LOG_N("instance_id=%d", instance_id);
    inst = (instance_t*)res_private(instance_id);

    /* save instance id */
    inst->instance_id = instance_id;

    /* setup arguments */
    strncpy(inst->paths.Local, local_path, PATH_MAX);
    strncpy(inst->paths.DocumentURL, SWF_PATH, PATH_MAX);
    strncpy(inst->paths.PluginInstanceURL, SWF_PATH "/" SWF_NAME "?" SWF_ARGS, PATH_MAX);
    inst->fps = 50;
    inst->width = 1920;
    inst->height = 1080;
    inst->is_full_screen = 0;
    inst->is_full_frame = 1;
    inst->pop_cuda_shmem_handle = pop_cuda_shmem_handle;
    inst->push_cuda_shmem_handle = push_cuda_shmem_handle;

    /* load module */
    r = mod_load(&mod, so_name);
    if(!r)
    {
LOG_N("mod->id=%d", mod->id);

        inst->message_loop_id = mod->interface.message_loop->Create(inst->instance_id);

LOG_N("inst->message_loop_id=%d", inst->message_loop_id);

        r = mod->interface.message_loop->AttachToCurrentThread(inst->message_loop_id);

LOG_N("msg_loop_interface->AttachToCurrentThread=%d", r);

        r = PPB_MessageLoop_main_thread = pthread_self();

LOG_N("mod->instance_interface->DidCreate.....");
        r = mod->interface.instance->DidCreate(inst->instance_id, 3, argn, argv);
LOG_N("mod->instance_interface->DidCreate=%d", r);

        mod->interface.instance_private =
            (struct PPP_Instance_Private_0_1*)mod->PPP_GetInterface(PPP_INSTANCE_PRIVATE_INTERFACE_0_1);
LOG_N("mod->interface.instance_private=%p", mod->interface.instance_private);
        if(mod->interface.instance_private && mod->interface.instance_private->GetInstanceObject)
        {
LOG_N("mod->interface.instance_private->GetInstanceObject=%p", mod->interface.instance_private->GetInstanceObject);
            inst->private_instance_object = mod->interface.instance_private->GetInstanceObject(inst->instance_id);
        }
        else
            inst->private_instance_object = PP_MakeUndefined();
LOG_N("inst->private_instance_object.type=%d", inst->private_instance_object.type);
LOG_N("inst->private_instance_object.value.as_id=%d", (int)inst->private_instance_object.value.as_id);

LOG_N("mod->instance_interface->DidChangeView...");
        mod->interface.instance->DidChangeView(inst->instance_id, inst->instance_id);
LOG_N("mod->instance_interface->DidChangeView DONE");

        if(inst->is_full_frame)
        {
            struct PP_Var str;

            int url_loader = mod->interface.url_loader->Create(inst->instance_id);

            int url_request_info = mod->interface.url_request_info->Create(inst->instance_id);

            str = VarFromUtf8_c(inst->paths.PluginInstanceURL);

            mod->interface.url_request_info->SetProperty(url_request_info, PP_URLREQUESTPROPERTY_URL, str);
            PPB_Var_Release(str);

            str = VarFromUtf8_c("GET");
            mod->interface.url_request_info->SetProperty(url_request_info, PP_URLREQUESTPROPERTY_METHOD, str);
            PPB_Var_Release(str);

            mod->interface.url_loader->Open(url_loader, url_request_info, PP_MakeCompletionCallback(f1, NULL));

LOG_N("mod->instance_interface->HandleDocumentLoad...");
            mod->interface.instance->HandleDocumentLoad(inst->instance_id, url_loader);
LOG_N("mod->instance_interface->HandleDocumentLoad DONE");

            res_release(url_loader);
            res_release(url_request_info);
        };

////        mod->instance_interface->DidChangeView(inst->instance_id, inst->instance_id);

        signal(SIGINT, sighandler);

LOG_N("mod->instance_interface->DidChangeView...");
        mod->interface.instance->DidChangeView(inst->instance_id, inst->instance_id);
LOG_N("mod->instance_interface->DidChangeView DONE");

LOG_N("will try to call dumb method");
        {
            struct PP_Var str;

            str = VarFromUtf8_c("toggle_play");

            mod->interface.var_depricated->Call(inst->private_instance_object, str, 0, NULL, NULL);

            PPB_Var_Release(str);
        };

LOG_N("demo sleep....");
sleep(1);
LOG_N("\n\n\n\n\n\n\n....sleeping done");

LOG_N("Run main loop...");
        r = mod->interface.message_loop->Run(inst->message_loop_id);
LOG_N("Exiting...");

        PPB_Var_Release(inst->private_instance_object);
LOG_PL;
        mod->interface.instance->DidDestroy(inst->instance_id);
LOG_PL;
        res_release(inst->message_loop_id);
LOG_PL;
        if(inst->graphics_id)
            res_release(inst->graphics_id);
LOG_PL;
        mod_release(&mod);
LOG_PL;
    };
LOG_PL;
    res_release(instance_id);
LOG_PL;
    res_end();
LOG_PL;


    {
        CUresult e;

        if(CUDA_SUCCESS != (e = cuCtxPushCurrent(shmCU_CTX)))
            LOG_E("cuCtxPushCurrent failed: %s", getCudaDrvErrorString(e));

        if(CUDA_SUCCESS != (e = cudaFree(shmPtr)))
            LOG_E("cudaFree failed: %s", getCudaDrvErrorString(e));

        if(CUDA_SUCCESS != (e = cuCtxDestroy(shmCU_CTX)))
            LOG_E("cuCtxDestroy failed: %s", getCudaDrvErrorString(e));
    };

    return 0;
};
