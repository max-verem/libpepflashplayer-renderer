#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_image_data.h>

#include "log.h"

static PP_ImageDataFormat GetNativeImageDataFormat(void)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsImageDataFormatSupported(PP_ImageDataFormat format)
{
    LOG_NP;
    return 0;
};

static PP_Resource Create(PP_Instance instance, PP_ImageDataFormat format,
    const struct PP_Size* size, PP_Bool init_to_zero)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsImageData(PP_Resource image_data)
{
    LOG_NP;
    return 0;
};

static PP_Bool Describe(PP_Resource image_data, struct PP_ImageDataDesc* desc)
{
    LOG_NP;
    return 0;
};

static void* Map(PP_Resource image_data)
{
    LOG_NP;
    return 0;
};

static void Unmap(PP_Resource image_data)
{
    LOG_NP;
    return 0;
};

struct PPB_ImageData_1_0 PPB_ImageData_1_0_instance =
{
    .GetNativeImageDataFormat = GetNativeImageDataFormat,
    .IsImageDataFormatSupported = IsImageDataFormatSupported,
    .Create = Create,
    .IsImageData = IsImageData,
    .Describe = Describe,
    .Map = Map,
    .Unmap = Unmap,
};
