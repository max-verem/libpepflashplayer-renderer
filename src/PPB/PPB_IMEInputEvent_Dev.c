#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_ime_input_event_dev.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance,
    PP_InputEvent_Type type,
    PP_TimeTicks time_stamp,
    struct PP_Var text,
    uint32_t segment_number,
    const uint32_t segment_offsets[],
    int32_t target_segment,
    uint32_t selection_start,
    uint32_t selection_end)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsIMEInputEvent(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static struct PP_Var GetText(PP_Resource ime_event)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static uint32_t GetSegmentNumber(PP_Resource ime_event)
{
    LOG_NP;
    return 0;
};

static uint32_t GetSegmentOffset(PP_Resource ime_event, uint32_t index)
{
    LOG_NP;
    return 0;
};

static int32_t GetTargetSegment(PP_Resource ime_event)
{
    LOG_NP;
    return 0;
};

static void GetSelection(PP_Resource ime_event, uint32_t* start, uint32_t* end)
{
    LOG_NP;
};

struct PPB_IMEInputEvent_Dev_0_2 PPB_IMEInputEvent_Dev_0_2_instance =
{
    .Create = Create,
    .IsIMEInputEvent = IsIMEInputEvent,
    .GetText = GetText,
    .GetSegmentNumber = GetSegmentNumber,
    .GetSegmentOffset = GetSegmentOffset,
    .GetTargetSegment = GetTargetSegment,
    .GetSelection = GetSelection,
};
