#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void BlitFramebufferEXT(PP_Resource context,
                             GLint srcX0,
                             GLint srcY0,
                             GLint srcX1,
                             GLint srcY1,
                             GLint dstX0,
                             GLint dstY0,
                             GLint dstX1,
                             GLint dstY1,
                             GLbitfield mask,
                             GLenum filter)
{
    LOG_NP;
};


struct PPB_OpenGLES2FramebufferBlit_1_0 PPB_OpenGLES2FramebufferBlit_1_0_instance =
{
    .BlitFramebufferEXT = BlitFramebufferEXT,
};
