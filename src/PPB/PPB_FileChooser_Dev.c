#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_file_chooser_dev.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance,
    PP_FileChooserMode_Dev mode, struct PP_Var accept_types)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFileChooser(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static int32_t Show(PP_Resource chooser, struct PP_ArrayOutput output,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

struct PPB_FileChooser_Dev_0_6 PPB_FileChooser_Dev_0_6_instance =
{
    .Create = Create,
    .IsFileChooser = IsFileChooser,
    .Show = Show
};
