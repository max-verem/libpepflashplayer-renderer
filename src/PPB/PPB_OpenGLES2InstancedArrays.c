#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void DrawArraysInstancedANGLE(PP_Resource context,
                                   GLenum mode,
                                   GLint first,
                                   GLsizei count,
                                   GLsizei primcount)
{
    LOG_NP;
};

static void DrawElementsInstancedANGLE(PP_Resource context,
                                     GLenum mode,
                                     GLsizei count,
                                     GLenum type,
                                     const void* indices,
                                     GLsizei primcount)
{
    LOG_NP;
};

static void VertexAttribDivisorANGLE(PP_Resource context,
                                   GLuint index,
                                   GLuint divisor)
{
    LOG_NP;
};

struct PPB_OpenGLES2InstancedArrays_1_0 PPB_OpenGLES2InstancedArrays_1_0_instance =
{
    .DrawArraysInstancedANGLE = DrawArraysInstancedANGLE,
    .DrawElementsInstancedANGLE = DrawElementsInstancedANGLE,
    .VertexAttribDivisorANGLE = VertexAttribDivisorANGLE,
};
