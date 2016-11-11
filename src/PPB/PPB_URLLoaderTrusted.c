#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/trusted/ppb_url_loader_trusted.h>

#include "log.h"

/**
 * Grant this URLLoader the capability to make unrestricted cross-origin
 * requests.
 */
static void GrantUniversalAccess(PP_Resource loader)
{
    LOG_TD;
};

/**
 * Registers that the given function will be called when the upload or
 * downloaded byte count has changed. This is not exposed on the untrusted
 * interface because it can be quite chatty and encourages people to write
 * feedback UIs that update as frequently as the progress updates.
 *
 * The other serious gotcha with this callback is that the callback must not
 * mutate the URL loader or cause it to be destroyed.
 *
 * However, the proxy layer needs this information to push to the other
 * process, so we expose it here. Only one callback can be set per URL
 * Loader. Setting to a NULL callback will disable it.
 */
static void RegisterStatusCallback(PP_Resource loader, PP_URLLoaderTrusted_StatusCallback cb)
{
    LOG_TD;
};


struct PPB_URLLoaderTrusted_0_3 PPB_URLLoaderTrusted_0_3_instance =
{
    .GrantUniversalAccess = GrantUniversalAccess,
    .RegisterStatusCallback = RegisterStatusCallback,
};
