#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_printing_dev.h>

#include "log.h"
#include "res.h"

extern struct PPB_Printing_Dev_0_7 PPB_Printing_Dev_0_7_instance;

/** Create a resource for accessing printing functionality.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 *
 * @return A <code>PP_Resource</code> containing the printing resource if
 * successful or 0 if it could not be created.
 */
static PP_Resource Create(PP_Instance instance)
{
    LOG("");
    res_add_ref(instance);
    return instance;
};

/**
 * Outputs the default print settings for the default printer into
 * <code>print_settings</code>. The callback is called with
 * <code>PP_OK</code> when the settings have been retrieved successfully.
 *
 * @param[in] resource The printing resource.
 *
 * @param[in] callback A <code>CompletionCallback</code> to be called when
 * <code>print_settings</code> have been retrieved.
 *
 * @return PP_OK_COMPLETIONPENDING if request for the default print settings
 * was successful, another error code from pp_errors.h on failure.
 */

static int32_t GetDefaultPrintSettings(PP_Resource resource,
    struct PP_PrintSettings_Dev* print_settings,
    struct PP_CompletionCallback callback)
{
    LOG_NP;

    return PP_OK_COMPLETIONPENDING;
};

struct PPB_Printing_Dev_0_7 PPB_Printing_Dev_0_7_instance =
{
    .Create = Create,
    .GetDefaultPrintSettings = GetDefaultPrintSettings,
};
