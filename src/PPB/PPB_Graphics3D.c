#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_graphics_3d.h>

#include "log.h"

static int32_t GetAttribMaxValue(PP_Resource instance, int32_t attribute, int32_t* value)
{
    LOG_NP;
    return 0;
};

static PP_Resource Create(PP_Instance instance, PP_Resource share_context, const int32_t attrib_list[])
{
    LOG_NP;
    return 0;
};

static PP_Bool IsGraphics3D(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static int32_t GetAttribs(PP_Resource context, int32_t attrib_list[])
{
    LOG_NP;
    return 0;
};

static int32_t SetAttribs(PP_Resource context, const int32_t attrib_list[])
{
    LOG_NP;
    return 0;
};

static int32_t GetError(PP_Resource context)
{
    LOG_NP;
    return 0;
};

static int32_t ResizeBuffers(PP_Resource context, int32_t width, int32_t height)
{
    LOG_NP;
    return 0;
};

static int32_t SwapBuffers(PP_Resource context, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

struct PPB_Graphics3D_1_0 PPB_Graphics3D_1_0_instance =
{
    .GetAttribMaxValue = GetAttribMaxValue,
    .Create = Create,
    .IsGraphics3D = IsGraphics3D,
    .GetAttribs = GetAttribs,
    .SetAttribs = SetAttribs,
    .GetError = GetError,
    .ResizeBuffers = ResizeBuffers,
    .SwapBuffers = SwapBuffers,
};
