#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>

#include <ppapi/c/ppb_core.h>

#include "impl/PPB_MessageLoop.h"

#include "log.h"

#include "res.h"

static void AddRefResource(PP_Resource resource)
{
    int r;

    r = res_add_ref(resource);

    LOG("res_add_ref(%d)=%d", resource, r);
};

static void ReleaseResource(PP_Resource resource)
{
    int r;

    r = res_release(resource);

    LOG("res_release(%d)=%d", resource, r);
};

static PP_Time GetTime(void)
{
    LOG_NP;
    return 0;
};

static PP_TimeTicks GetTimeTicks(void)
{
    LOG_NP;
    return 0;
};

static void CallOnMainThread(int32_t delay_in_milliseconds, struct PP_CompletionCallback callback, int32_t result)
{
    LOG_NP;
    LOG("delay_in_milliseconds=%d, result=%d", delay_in_milliseconds, result);
};

static PP_Bool IsMainThread(void)
{
    return (pthread_self() == PPB_MessageLoop_main_thread) ? PP_TRUE : PP_FALSE;
};

struct PPB_Core_1_0 PPB_Core_1_0_interface =
{
    .AddRefResource = AddRefResource,
    .ReleaseResource = ReleaseResource,
    .GetTime = GetTime,
    .GetTimeTicks = GetTimeTicks,
    .CallOnMainThread = CallOnMainThread,
    .IsMainThread = IsMainThread,
};
