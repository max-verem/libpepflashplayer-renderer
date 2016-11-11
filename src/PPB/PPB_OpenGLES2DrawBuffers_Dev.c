#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_opengles2ext_dev.h>

#include "log.h"

static void DrawBuffersEXT(PP_Resource context,
                         GLsizei count,
                         const GLenum* bufs)
{
    LOG_NP;
};

struct PPB_OpenGLES2DrawBuffers_Dev_1_0 PPB_OpenGLES2DrawBuffers_Dev_1_0_instance =
{
    .DrawBuffersEXT = DrawBuffersEXT,
};
