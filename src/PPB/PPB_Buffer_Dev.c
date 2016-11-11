#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_buffer_dev.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance, uint32_t size_in_bytes)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsBuffer(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static PP_Bool Describe(PP_Resource resource, uint32_t* size_in_bytes)
{
    LOG_NP;
    return 0;
};

static void* Map(PP_Resource resource)
{
    LOG_NP;
    return NULL;
};

static void Unmap(PP_Resource resource)
{
    LOG_NP;
};

struct PPB_Buffer_Dev_0_4 PPB_Buffer_Dev_0_4_instance =
{
    .Create = Create,
    .IsBuffer = IsBuffer,
    .Describe = Describe,
    .Map = Map,
    .Unmap = Unmap,
};
