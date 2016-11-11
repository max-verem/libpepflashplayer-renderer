#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_memory_dev.h>

#include "log.h"

void* MemAlloc(uint32_t num_bytes)
{
    return malloc(num_bytes);
};

void MemFree(void* ptr)
{
    free(ptr);
};

struct PPB_Memory_Dev_0_1 PPB_Memory_Dev_0_1_instance =
{
    .MemAlloc = MemAlloc,
    .MemFree = MemFree,
};

