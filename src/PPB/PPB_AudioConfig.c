#include <stdint.h>
#include <stdio.h>
#include <string.h>

//#include <ppapi/c/pp_errors.h>
//#include <ppapi/c/pp_time.h>
//#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>

//#include <ppapi/c/ppb_core.h>
#include <ppapi/c/ppb_audio_config.h>

#include "log.h"
#include "res.h"

struct PPB_AudioConfig_1_1 PPB_AudioConfig_1_1_instance;

typedef struct
{
    PP_AudioSampleRate sample_rate;
    uint32_t sample_frame_count;
} audio_config_t;

static void Destructor(audio_config_t* ctx)
{
};

static PP_Resource CreateStereo16Bit(PP_Instance instance,
    PP_AudioSampleRate sample_rate, uint32_t sample_frame_count)
{
    audio_config_t* ctx;
    PP_Resource res = res_create(sizeof(audio_config_t), &PPB_AudioConfig_1_1_instance, (res_destructor_t)Destructor);

    ctx = (audio_config_t*)res_private(res);
    ctx->sample_rate = sample_rate;
    ctx->sample_frame_count = sample_frame_count;

    LOG("sample_rate=%d, sample_frame_count=%d", sample_rate, sample_frame_count);

    return res;
};

static uint32_t RecommendSampleFrameCount(PP_Instance instance,
    PP_AudioSampleRate sample_rate, uint32_t requested_sample_frame_count)
{
    LOG_TD;

    return PP_AUDIOSAMPLERATE_48000 / 25;
};

static PP_Bool IsAudioConfig(PP_Resource resource)
{
    LOG_NP;

    return 1;
};

static PP_AudioSampleRate GetSampleRate(PP_Resource config)
{
    LOG_NP;

    return PP_AUDIOSAMPLERATE_48000;
};

static uint32_t GetSampleFrameCount(PP_Resource config)
{
    LOG_NP;

    return PP_AUDIOSAMPLERATE_48000 / 25;
};

static PP_AudioSampleRate RecommendSampleRate(PP_Instance instance)
{
    LOG_NP;

    return PP_AUDIOSAMPLERATE_48000;
};

struct PPB_AudioConfig_1_1 PPB_AudioConfig_1_1_instance =
{
    .CreateStereo16Bit = CreateStereo16Bit,
    .RecommendSampleFrameCount = RecommendSampleFrameCount,
    .IsAudioConfig = IsAudioConfig,
    .GetSampleRate = GetSampleRate,
    .GetSampleFrameCount = GetSampleFrameCount,
    .RecommendSampleRate = RecommendSampleRate,
};
