#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void RenderbufferStorageMultisampleEXT(PP_Resource context,
                                            GLenum target,
                                            GLsizei samples,
                                            GLenum internalformat,
                                            GLsizei width,
                                            GLsizei height)
{
    LOG_NP;
};

struct PPB_OpenGLES2FramebufferMultisample_1_0 PPB_OpenGLES2FramebufferMultisample_1_0_instance =
{
    .RenderbufferStorageMultisampleEXT = RenderbufferStorageMultisampleEXT,
};
