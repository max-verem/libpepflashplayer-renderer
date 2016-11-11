#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_cursor_control_dev.h>

#include "log.h"

static PP_Bool SetCursor(PP_Instance instance,
    enum PP_CursorType_Dev type, PP_Resource custom_image,
    const struct PP_Point* hot_spot)
{
    LOG_NP;
    return 0;
};

static PP_Bool LockCursor(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

static PP_Bool UnlockCursor(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

static PP_Bool HasCursorLock(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

static PP_Bool CanLockCursor(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

struct PPB_CursorControl_Dev_0_4 PPB_CursorControl_Dev_0_4_instance =
{
    .SetCursor = SetCursor,
    .LockCursor = LockCursor,
    .UnlockCursor = UnlockCursor,
    .HasCursorLock = HasCursorLock,
    .CanLockCursor = CanLockCursor,
};
