#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "log.h"
#include "mod.h"
#include "if.h"
#include "res.h"

static void mod_destructor(void* p)
{
    mod_t* mod = (mod_t*)p;

    if(mod->handle)
        dlclose(mod->handle);
};

int mod_load(mod_t** pmod, const char* so_name)
{
    int r = 0;

    int mod_id = res_create(sizeof(mod_t), NULL, mod_destructor);
    mod_t* mod = (mod_t*)res_private(mod_id);

    *pmod = NULL;

    mod->id = mod_id;

    mod->handle = dlopen(so_name, RTLD_NOW | RTLD_GLOBAL);
    if(mod->handle)
    {
        mod->PPP_InitializeModule = dlsym(mod->handle, "PPP_InitializeModule");
        mod->PPP_GetInterface = dlsym(mod->handle, "PPP_GetInterface");

        if(mod->PPP_InitializeModule && mod->PPP_GetInterface)
        {
            r = mod->PPP_InitializeModule(mod_id, get_browser_interface_proc);
            if(!r)
            {
                mod->interface.instance = mod->PPP_GetInterface(PPP_INSTANCE_INTERFACE_1_1);
                if(mod->interface.instance)
                {
                    *pmod = mod;

                    /* load interfaces */

                    mod->interface.message_loop =
                        (struct PPB_MessageLoop_1_0*)if_find(PPB_MESSAGELOOP_INTERFACE_1_0)->ptr;

                    mod->interface.instance_private =
                        (struct PPP_Instance_Private_0_1*)mod->PPP_GetInterface(PPP_INSTANCE_PRIVATE_INTERFACE_0_1);

                    mod->interface.url_loader =
                        (struct PPB_URLLoader_1_0*)if_find(PPB_URLLOADER_INTERFACE_1_0)->ptr;

                    mod->interface.url_request_info =
                        (struct PPB_URLRequestInfo_1_0*)if_find(PPB_URLREQUESTINFO_INTERFACE_1_0)->ptr;

                    return 0;
                }
                else
                {
                    LOG_E("PPP_GetInterface(%s) failed", PPP_INSTANCE_INTERFACE_1_1);
                    r = -ENOENT;
                };
            }
            else
                LOG_E("PPP_InitializeModule failed, r=%d", r);
        }
        else
        {
            if(!mod->PPP_InitializeModule)
                LOG_E("dlopen(%s) failed", "PPP_InitializeModule");

            if(!mod->PPP_GetInterface)
                LOG_E("dlopen(%s) failed", "PPP_GetInterface");

            r = -ENOENT;
        };
    }
    else
    {
        r = -errno;
        LOG_E("dlopen(%s) failed, r=%d", so_name, r);
    };

    res_release(mod->id);

    return r;
};

int mod_release(mod_t** pmod)
{
    mod_t* mod;

    if(!pmod)
        return -EINVAL;

    mod = *pmod;

    if(!mod)
        return -EINVAL;

    *pmod = NULL;

    res_release(mod->id);

    return 0;
};
