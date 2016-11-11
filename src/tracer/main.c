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

#include "impl/PPB_MessageLoop.h"

#define SWF "demo.swf"
const char* so_name = "/usr/local/src/2016-10-28/libpepflashplayer.so";
const char* so_path = "/usr/local/src/2016-10-28/demos/.Local";
const char* swf_path = "file:///usr/local/src/2016-10-28/demos";
const char* swf_name = SWF;

const char* argn[] = { "src", "quality", "bgcolor", "width", "height", NULL};
const char* argv[] = { SWF,   "high",    "#ece9d8",  "1920", "1080", NULL};


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
    strncpy(inst->paths.Local, so_path, PATH_MAX);
    strncpy(inst->paths.DocumentURL, swf_path, PATH_MAX);
    strncpy(inst->paths.PluginInstanceURL, swf_path, PATH_MAX);
    strncat(inst->paths.PluginInstanceURL, "/", PATH_MAX);
    strncat(inst->paths.PluginInstanceURL, swf_name, PATH_MAX);

    r = mod_load(so_name, &mod);
    if(!r)
    {
        struct PPB_MessageLoop_1_0* msg_loop_interface =
            (struct PPB_MessageLoop_1_0*)if_find(PPB_MESSAGELOOP_INTERFACE_1_0)->ptr;

LOG("mod->id=%d", mod->id);

#if 1
        inst->message_loop_id = msg_loop_interface->Create(inst->instance_id);

LOG("inst->message_loop_id=%d", inst->message_loop_id);

        r = msg_loop_interface->AttachToCurrentThread(inst->message_loop_id);

LOG("msg_loop_interface->AttachToCurrentThread=%d", r);

        r = PPB_MessageLoop_main_thread = pthread_self();
#endif

        r = mod->instance_interface->DidCreate(inst->instance_id, 5, argn, argv);

LOG("mod->instance_interface->DidCreate=%d", r);

        mod->instance_private_interface = (struct PPP_Instance_Private_0_1*)mod->PPP_GetInterface(PPP_INSTANCE_PRIVATE_INTERFACE_0_1);
        if(mod->instance_private_interface && mod->instance_private_interface->GetInstanceObject)
        {
            inst->private_instance_object = mod->instance_private_interface->GetInstanceObject(inst->instance_id);
        }
        else
            inst->private_instance_object = PP_MakeUndefined();

LOG("mod->instance_private_interface=%p", mod->instance_private_interface);

LOG("Sleeping...");
sleep(3);
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
