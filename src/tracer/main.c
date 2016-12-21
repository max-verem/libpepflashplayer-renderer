#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "log.h"
#include "mod.h"
#include "res.h"
#include "instance.h"
#include "if.h"

#include <ppapi/c/ppb_message_loop.h>
#include <ppapi/c/ppb_url_loader.h>
#include <ppapi/c/ppb_url_request_info.h>
#include <ppapi/c/ppb_view.h>

#include "PPB_MessageLoop.h"
#include "PPB_Var.h"

#define SWF "TST-dyn_text-faded.swf"
//#define SWF "1080i50-blank_test_GPU.swf"
//#define SWF "m1_logo_1080i50_BIG.swf"
//#define SWF "m1_logo_1080i50.swf"
//#define SWF "demo4.swf"
//#define SWF "demo1_2_movie.swf"
//#define SWF "demo1_2_image.swf"
//#define SWF "demo2.swf"

const char* so_name = "/usr/local/src/libpepflashplayer-renderer.git/tests/libpepflashplayer.so-24.0.0.170";
const char* local_path = "/usr/local/src/libpepflashplayer-renderer.git/tests/local";
const char* swf_path = "file:///usr/local/src/libpepflashplayer-renderer.git/tests/src";
const char* swf_name = SWF;

const char* argn[] = { /* "src", */ "quality", "bgcolor", "width", "height", "wmode", /* "flashvars", */ NULL};
const char* argv[] = { /* SWF, */ "high",    "#ece9d8",  "1920", "1080", "transparent", /* "line1=foo_flashvars&line2=bar_flashvars", */ NULL};

static void f1(void* user_data, int32_t result)
{
    LOG_TD;
};

int main()
{
    int r, instance_id;
    mod_t* mod;
    instance_t* inst;

//    struct PP_URLComponents_Dev comp;
//    uriparser_parse("http://root:@demo1.com:1232/ho/ms/com?bla=123#paper", &comp);
//    return 0;

    res_begin();

    instance_id = res_create(sizeof(instance_t), NULL, NULL);
LOG("instance_id=%d", instance_id);
    inst = (instance_t*)res_private(instance_id);

    /* save instance id */
    inst->instance_id = instance_id;

    /* setup arguments */
    strncpy(inst->paths.Local, local_path, PATH_MAX);
    strncpy(inst->paths.DocumentURL, swf_path, PATH_MAX);
    strncpy(inst->paths.PluginInstanceURL, swf_path, PATH_MAX);
    strncat(inst->paths.PluginInstanceURL, "/", PATH_MAX);
    strncat(inst->paths.PluginInstanceURL, swf_name, PATH_MAX);

    strncat(inst->paths.PluginInstanceURL, "?line1=foo_url1&line2=bar_url2&line3=foo_url3&line4=bar_url4", PATH_MAX);

//    inst->is_full_screen = 1;
    inst->is_full_frame = 1;

    r = mod_load(so_name, &mod);
    if(!r)
    {
        struct PPB_MessageLoop_1_0* msg_loop_interface =
            (struct PPB_MessageLoop_1_0*)if_find(PPB_MESSAGELOOP_INTERFACE_1_0)->ptr;


LOG("mod->id=%d", mod->id);

        inst->message_loop_id = msg_loop_interface->Create(inst->instance_id);

LOG("inst->message_loop_id=%d", inst->message_loop_id);

        r = msg_loop_interface->AttachToCurrentThread(inst->message_loop_id);

LOG("msg_loop_interface->AttachToCurrentThread=%d", r);

        r = PPB_MessageLoop_main_thread = pthread_self();

        mod->instance_interface->DidChangeView(inst->instance_id, inst->instance_id);
LOG("mod->instance_interface->DidChangeView");

        r = mod->instance_interface->DidCreate(inst->instance_id, 6, argn, argv);

LOG("mod->instance_interface->DidCreate=%d", r);

        mod->instance_private_interface = (struct PPP_Instance_Private_0_1*)mod->PPP_GetInterface(PPP_INSTANCE_PRIVATE_INTERFACE_0_1);
        if(mod->instance_private_interface && mod->instance_private_interface->GetInstanceObject)
        {
            inst->private_instance_object = mod->instance_private_interface->GetInstanceObject(inst->instance_id);
        }
        else
            inst->private_instance_object = PP_MakeUndefined();

LOG("mod->instance_private_interface=%p", mod->instance_private_interface);

LOG("mod->instance_interface->DidChangeView...");

        mod->instance_interface->DidChangeView(inst->instance_id, inst->instance_id);


//        mod->instance_interface->DidChangeFocus(inst->instance_id, 1);

LOG("mod->instance_interface->DidChangeView DONE");

        if(inst->is_full_frame)
        {
            struct PP_Var str;

            struct PPB_URLLoader_1_0* url_loader_interface =
                (struct PPB_URLLoader_1_0*)if_find(PPB_URLLOADER_INTERFACE_1_0)->ptr;

            struct PPB_URLRequestInfo_1_0* url_request_info_interface =
                (struct PPB_URLRequestInfo_1_0*)if_find(PPB_URLREQUESTINFO_INTERFACE_1_0)->ptr;


            int url_loader = url_loader_interface->Create(inst->instance_id);

            int url_request_info = url_request_info_interface->Create(inst->instance_id);

            str = VarFromUtf8_c(inst->paths.PluginInstanceURL);
            url_request_info_interface->SetProperty(url_request_info, PP_URLREQUESTPROPERTY_URL, str);
            PPB_Var_Release(str);

            str = VarFromUtf8_c("GET");
            url_request_info_interface->SetProperty(url_request_info, PP_URLREQUESTPROPERTY_METHOD, str);
            PPB_Var_Release(str);

            url_loader_interface->Open(url_loader, url_request_info, PP_MakeCompletionCallback(f1, NULL));

LOG("mod->instance_interface->HandleDocumentLoad...");
            mod->instance_interface->HandleDocumentLoad(inst->instance_id, url_loader);
LOG("mod->instance_interface->HandleDocumentLoad DONE");

            res_release(url_loader);
            res_release(url_request_info);
        };

////        mod->instance_interface->DidChangeView(inst->instance_id, inst->instance_id);

LOG("Run main loop...");
        r = msg_loop_interface->Run(inst->message_loop_id);
LOG("Exiting...");

        mod->instance_interface->DidDestroy(inst->instance_id);
LOG("");
        res_release(inst->message_loop_id);
LOG("");
        mod_release(&mod);
LOG("");

    };
LOG("");
    res_release(instance_id);
LOG("");
    res_end();
LOG("");

    return 0;
};
