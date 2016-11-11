#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/trusted/ppb_file_chooser_trusted.h>

#include "log.h"

int32_t ShowWithoutUserGesture(PP_Resource chooser, PP_Bool save_as,
    struct PP_Var suggested_file_name, struct PP_ArrayOutput output,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

struct PPB_FileChooserTrusted_0_6 PPB_FileChooserTrusted_0_6_instance =
{
    .ShowWithoutUserGesture = ShowWithoutUserGesture,
};
