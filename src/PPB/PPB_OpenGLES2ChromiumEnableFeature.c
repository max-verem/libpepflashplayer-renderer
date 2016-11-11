#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static GLboolean EnableFeatureCHROMIUM(PP_Resource context, const char* feature)
{
    LOG_NP;
    return 0;
};

struct PPB_OpenGLES2ChromiumEnableFeature_1_0 PPB_OpenGLES2ChromiumEnableFeature_1_0_instance =
{
    .EnableFeatureCHROMIUM = EnableFeatureCHROMIUM,
};
