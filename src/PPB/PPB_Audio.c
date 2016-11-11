#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_audio.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance,
    PP_Resource config, PPB_Audio_Callback audio_callback, void* user_data)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsAudio(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static PP_Resource GetCurrentConfig(PP_Resource audio)
{
    LOG_NP;
    return 0;
};

static PP_Bool StartPlayback(PP_Resource audio)
{
    LOG_NP;
    return 0;
};

static PP_Bool StopPlayback(PP_Resource audio)
{
    LOG_NP;
    return 0;
};

struct PPB_Audio_1_1 PPB_Audio_1_1_instance =
{
    .Create = Create,
    .IsAudio = IsAudio,
    .GetCurrentConfig = GetCurrentConfig,
    .StartPlayback = StartPlayback,
    .StopPlayback = StopPlayback,
};
