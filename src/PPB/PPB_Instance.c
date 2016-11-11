#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_instance.h>

#include "log.h"
#include "instance.h"
#include "res.h"

static PP_Bool BindGraphics(PP_Instance instance, PP_Resource device)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFullFrame(PP_Instance instance)
{
    instance_t* inst = (instance_t*)res_private(instance);
    return inst->is_full_frame;
};

struct PPB_Instance_1_0 PPB_Instance_1_0_instance =
{
    .BindGraphics = BindGraphics,
    .IsFullFrame = IsFullFrame,
};
