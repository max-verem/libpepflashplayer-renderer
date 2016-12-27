#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_network_monitor.h>

#include "log.h"
#include "res.h"

/**
 * Creates a Network Monitor resource.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance of
 * a module.
 *
 * @return A <code>PP_Resource</code> corresponding to a network monitor or 0
 * on failure.
 */
static PP_Resource Create(PP_Instance instance)
{
    LOG("");
    res_add_ref(instance);
    return instance;
};


/**
 * Gets current network configuration. When called for the first time,
 * completes as soon as the current network configuration is received from
 * the browser. Each consequent call will wait for network list changes,
 * returning a new <code>PPB_NetworkList</code> resource every time.
 *
 * @param[in] network_monitor A <code>PP_Resource</code> corresponding to a
 * network monitor.
 * @param[out] network_list The <code>PPB_NetworkList<code> resource with the
 * current state of network interfaces.
 * @param[in] callback A <code>PP_CompletionCallback</code> to be called upon
 * completion.
 *
 * @return An int32_t containing an error code from <code>pp_errors.h</code>.
 * <code>PP_ERROR_NOACCESS</code> will be returned if the caller doesn't have
 * required permissions.
 */
static int32_t UpdateNetworkList(PP_Resource network_monitor,
    PP_Resource* network_list, struct PP_CompletionCallback callback)
{
    LOG_TD;
    return PP_OK_COMPLETIONPENDING;
};


/**
 * Determines if the specified <code>resource</code> is a
 * <code>NetworkMonitor</code> object.
 *
 * @param[in] resource A <code>PP_Resource</code> resource.
 *
 * @return Returns <code>PP_TRUE</code> if <code>resource</code> is a
 * <code>PPB_NetworkMonitor</code>, <code>PP_FALSE</code>otherwise.
 */
static PP_Bool IsNetworkMonitor(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

/**
 * The <code>PPB_NetworkMonitor</code> allows to get network interfaces
 * configuration and monitor network configuration changes.
 *
 * Permissions: Apps permission <code>socket</code> with subrule
 * <code>network-state</code> is required for <code>UpdateNetworkList()</code>.
 * For more details about network communication permissions, please see:
 * http://developer.chrome.com/apps/app_network.html
 */
struct PPB_NetworkMonitor_1_0 PPB_NetworkMonitor_1_0_instance =
{
    .Create = Create,
    .UpdateNetworkList = UpdateNetworkList,
    .IsNetworkMonitor = IsNetworkMonitor,
};
