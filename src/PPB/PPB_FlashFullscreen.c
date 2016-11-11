#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash_fullscreen.h>

#include "log.h"
#include "instance.h"
#include "res.h"

static PP_Bool IsFullscreen(PP_Instance instance)
{
    instance_t* inst = (instance_t*)res_private(instance);
    return inst->is_full_screen;
};

static PP_Bool SetFullscreen(PP_Instance instance, PP_Bool fullscreen)
{
    LOG("fullscreen=%d", fullscreen);
    return 1;
};

static PP_Bool GetScreenSize(PP_Instance instance, struct PP_Size* size)
{
    LOG_TD;
    size->width = 1920;
    size->height = 1080;
    return 1;
};

struct PPB_FlashFullscreen_1_0 PB_FlashFullscreen_1_0_instance =
{
    .IsFullscreen = IsFullscreen,
    .SetFullscreen = SetFullscreen,
    .GetScreenSize = GetScreenSize,
};
