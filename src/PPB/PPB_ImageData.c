#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_image_data.h>

#include "log.h"
#include "res.h"

#include "PPB_ImageData.h"

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

struct PPB_ImageData_1_0 PPB_ImageData_1_0_instance;

static void Destructor(image_data_t* ctx)
{
    LOG("{%d}", ctx->self);
};

static PP_Resource Create(PP_Instance instance, PP_ImageDataFormat format,
    const struct PP_Size* size, PP_Bool init_to_zero)
{
    int res = res_create(sizeof(image_data_t), &PPB_ImageData_1_0_instance, (res_destructor_t)Destructor);

    image_data_t* image_data = (image_data_t*)res_private(res);

    image_data->instance_id = instance;
    image_data->self = res;
    image_data->size = *size;
    image_data->format = format;
    image_data->buf = calloc(1, 4 * size->width * size->height);

    LOG("res=%d, size->width=%d, size->height=%d", res, size->width, size->height);

    return res;
};

static PP_Bool IsImageData(PP_Resource image_data)
{
    return (res_interface(image_data) == &PPB_ImageData_1_0_instance);
};

static PP_Bool Describe(PP_Resource res, struct PP_ImageDataDesc* desc)
{
    image_data_t* image_data = (image_data_t*)res_private(res);

    desc->size = image_data->size;
    desc->format = image_data->format;
    desc->stride = 4 * image_data->size.width;

    return 1;
};

static void* Map(PP_Resource res)
{
    image_data_t* image_data = (image_data_t*)res_private(res);

    return image_data->buf;
};

static void Unmap(PP_Resource image_data)
{
    LOG_TD;
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
