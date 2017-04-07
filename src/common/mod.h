#ifndef MODULE_H
#define MODULE_H

#include <stdint.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppb.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_errors.h>

#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/private/ppp_instance_private.h>
#include <ppapi/c/ppb_message_loop.h>
#include <ppapi/c/ppb_url_loader.h>
#include <ppapi/c/ppb_url_request_info.h>
#include <ppapi/c/dev/ppb_var_deprecated.h>
#include <ppapi/c/ppb_view.h>

typedef struct mod_desc
{
    int id;
    void *handle;
    int32_t (*PPP_InitializeModule)(PP_Module module, PPB_GetInterface get_browser_interface);
    const void * (*PPP_GetInterface)(const char* interface_name);
    struct
    {
        const struct PPP_Instance_1_1* instance;
        const struct PPP_Instance_Private_0_1* instance_private;
    } interface;
} mod_t;

int mod_load(mod_t** pmod, const char* so_name);
int mod_release(mod_t** pmod);

#endif
