#include <dlfcn.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>

#include "log.h"
#include "mod.h"
#include "res.h"
#include "instance.h"
#include "if.h"

#include "PPB_MessageLoop.h"
#include "PPB_Var.h"
#include "PPB.h"

#define SWF_PATH "file:///usr/local/src/libpepflashplayer-renderer.git/tests/src"

/*
#define SWF_ARGS "line0=This%20String%20in%20File%20[" __FILE__ "]%20"
#define SWF_NAME "titleSD_full_screen.swf"
*/

#define SWF_ARGS ""
#define SWF_NAME "CtlProxy.swf"

//const char* so_name = "/usr/local/src/libpepflashplayer-renderer.git/tests/libpepflashplayer.so-24.0.0.186-debug";
const char* so_name = "/usr/local/src/libpepflashplayer-renderer.git/tests/libpepflashplayer.so-25.0.0.143-debug";
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
        r = PPB_MessageLoop_1_0_instance.PostQuit(inst->message_loop_id, 0);

    LOG_N("r=%d", r);
};

#include "app.h"

int main()
{
    app_t* app_data;
    int r, instance_id;

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

    app_create(&app_data, inst);

    /* load module */
    r = mod_load(&mod, so_name);
    if(!r)
    {
        inst->window_instance_object = PPB_Var_Deprecated_instance.CreateObject(inst->instance_id, inst->app_class, inst->app_data);

LOG_N("mod->id=%d", mod->id);

        inst->message_loop_id = PPB_MessageLoop_1_0_instance.Create(inst->instance_id);

LOG_N("inst->message_loop_id=%d", inst->message_loop_id);

        r = PPB_MessageLoop_1_0_instance.AttachToCurrentThread(inst->message_loop_id);

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

            int url_loader = PPB_URLLoader_1_0_instance.Create(inst->instance_id);

            int url_request_info = PPB_URLRequestInfo_1_0_instance.Create(inst->instance_id);

            str = VarFromUtf8_c(inst->paths.PluginInstanceURL);

            PPB_URLRequestInfo_1_0_instance.SetProperty(url_request_info, PP_URLREQUESTPROPERTY_URL, str);
            PPB_Var_Release(str);

            str = VarFromUtf8_c("GET");
            PPB_URLRequestInfo_1_0_instance.SetProperty(url_request_info, PP_URLREQUESTPROPERTY_METHOD, str);
            PPB_Var_Release(str);

            PPB_URLLoader_1_0_instance.Open(url_loader, url_request_info, PP_MakeCompletionCallback(f1, NULL));

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

LOG_N("demo sleep1....");
sleep(1);
LOG_N("\n\n\n\n\n\n\n....sleeping done");

#if 0
LOG_N("will try to call dumb method");
        {
            struct PP_Var result, ex, method;
            struct PP_Var argv[3];

            LOG_N("will create obj");
            method = VarFromUtf8_c("toggle_play");
            argv[0] = VarFromUtf8_c("fooo_arg");
            argv[1] = PP_MakeInt32(12345);
//            argv[1] = PPB_VarArrayBuffer_1_0_instance.Create(128);
//            argv[1] = PPB_VarArray_1_0_instance.Create();
//            argv[1] = PPB_Instance_Private_0_1_instance.GetWindowObject(inst->instance_id);
//            arg = PPB_Var_Deprecated_instance.CreateObject(inst->instance_id, inst->app_class, inst->app_data);
            LOG_N("done create obj");

            LOG_N("will call toggle_play");
            result = PPB_Var_Deprecated_instance.Call(inst->private_instance_object, method, 3, argv, &ex);
            PPB_Var_Dump("toggle_play result", result);
            PPB_Var_Dump("toggle_play ex", result);
            PPB_Var_Dump("toggle_play arg2", argv[2]);

            PPB_Var_Release(ex);
            PPB_Var_Release(method);
            PPB_Var_Release(argv[0]);
            PPB_Var_Release(argv[1]);
            PPB_Var_Release(result);
        };
#endif

LOG_N("will run reader thread");
        app_run(app_data);


LOG_N("demo sleep2....");
sleep(1);
LOG_N("\n\n\n\n\n\n\n....sleeping done");

LOG_N("Run main loop...");
        r = PPB_MessageLoop_1_0_instance.Run(inst->message_loop_id);
LOG_N("Exiting...");


LOG_N("will STOP reader thread");
        app_stop(app_data);

LOG_PL;
        mod->interface.instance->DidDestroy(inst->instance_id);

        PPB_Var_Release(inst->private_instance_object);
        PPB_Var_Release(inst->window_instance_object);
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
    app_destroy(&app_data);
LOG_PL;
    res_release(instance_id);
LOG_PL;
    res_end();
LOG_PL;

    return 0;
};
