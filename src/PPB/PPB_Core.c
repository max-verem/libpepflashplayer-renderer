#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>

#include <ppapi/c/ppb_core.h>

#include "PPB_MessageLoop.h"

#include "log.h"

#include "res.h"

static void AddRefResource(PP_Resource resource)
{
    int r;

    r = res_add_ref(resource);

    LOG1("res_add_ref(%d)=%d", resource, r);
};

static void ReleaseResource(PP_Resource resource)
{
    int r;

    r = res_release(resource);

    LOG1("res_release(%d)=%d", resource, r);
};

/**
 * GetTime() returns the "wall clock time" according to the
 * browser.
 *
 * @return A <code>PP_Time</code> containing the "wall clock time" according
 * to the browser.
 */
static PP_Time GetTime(void)
{
    time_t t;
    time(&t);
    return t;
};

/**
 * GetTimeTicks() returns the "tick time" according to the browser.
 * This clock is used by the browser when passing some event times to the
 * module (e.g. using the <code>PP_InputEvent::time_stamp_seconds</code>
 * field). It is not correlated to any actual wall clock time
 * (like GetTime()). Because of this, it will not run change if the user
 * changes their computer clock.
 *
 * @return A <code>PP_TimeTicks</code> containing the "tick time" according
 * to the browser.
 */
static PP_TimeTicks GetTimeTicks(void)
{
    LOG_NP;
    return 0;
};

/**
 * CallOnMainThread() schedules work to be executed on the main module thread
 * after the specified delay. The delay may be 0 to specify a call back as
 * soon as possible.
 *
 * The <code>result</code> parameter will just be passed as the second
 * argument to the callback. Many applications won't need this, but it allows
 * a module to emulate calls of some callbacks which do use this value.
 *
 * <strong>Note:</strong> CallOnMainThread, even when used from the main
 * thread with a delay of 0 milliseconds, will never directly invoke the
 * callback.Even in this case, the callback will be scheduled
 * asynchronously.
 *
 * <strong>Note:</strong> If the browser is shutting down or if the module
 * has no instances, then the callback function may not be called.
 *
 * @param[in] delay_in_milliseconds An int32_t delay in milliseconds.
 * @param[in] callback A <code>PP_CompletionCallback</code> callback function
 * that the browser will call after the specified delay.
 * @param[in] result An int32_t that the browser will pass to the given
 * <code>PP_CompletionCallback</code>.
 */
static void CallOnMainThread(int32_t delay_in_milliseconds, struct PP_CompletionCallback callback, int32_t result)
{
    LOG1("");
    PPB_MessageLoop_push(0, callback, delay_in_milliseconds, result);
};

/**
 * IsMainThread() returns true if the current thread is the main pepper
 * thread.
 *
 * This function is useful for implementing sanity checks, and deciding if
 * dispatching using CallOnMainThread() is required.
 *
 * @return A <code>PP_Bool</code> containing <code>PP_TRUE</code> if the
 * current thread is the main pepper thread, otherwise <code>PP_FALSE</code>.
 */
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
