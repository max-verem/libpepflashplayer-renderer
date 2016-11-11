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

int mod_load(const char* so_name, mod_t** pmod)
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
                mod->instance_interface = mod->PPP_GetInterface(PPP_INSTANCE_INTERFACE_1_1);
                if(mod->instance_interface)
                {
                    *pmod = mod;
                    return 0;
                }
                else
                {
                    LOG("PPP_GetInterface(%s) failed", PPP_INSTANCE_INTERFACE_1_1);
                    r = -ENOENT;
                };
            }
            else
                LOG("PPP_InitializeModule failed, r=%d", r);
        }
        else
        {
            if(!mod->PPP_InitializeModule)
                LOG("dlopen(%s) failed", "PPP_InitializeModule");

            if(!mod->PPP_GetInterface)
                LOG("dlopen(%s) failed", "PPP_GetInterface");

            r = -ENOENT;
        };
    }
    else
    {
        r = -errno;
        LOG("dlopen(%s) failed, r=%d", so_name, r);
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
