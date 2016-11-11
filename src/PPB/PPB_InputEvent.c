#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_input_event.h>

#include "log.h"

static int32_t RequestFilteringInputEvents(PP_Instance instance, uint32_t event_classes)
{
    return 0;
};

static int32_t RequestInputEvents(PP_Instance instance, uint32_t event_classes)
{
    return 0;
};

static void ClearInputEventRequest(PP_Instance instance, uint32_t event_classes)
{
};

static PP_Bool IsInputEvent(PP_Resource resource)
{
    return 0;
};

static PP_InputEvent_Type GetType(PP_Resource event)
{
    return 0;
};

static PP_TimeTicks GetTimeStamp(PP_Resource event)
{
    return 0;
};

static uint32_t GetModifiers(PP_Resource event)
{
    return 0;
};

struct PPB_InputEvent_1_0 PPB_InputEvent_1_0_instance =
{
    .RequestFilteringInputEvents = RequestFilteringInputEvents,
    .RequestInputEvents = RequestInputEvents,
    .ClearInputEventRequest = ClearInputEventRequest,
    .IsInputEvent = IsInputEvent,
    .GetType = GetType,
    .GetTimeStamp = GetTimeStamp,
    .GetModifiers = GetModifiers,
};
