#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void GenVertexArraysOES(PP_Resource context, GLsizei n, GLuint* arrays)
{
    LOG_NP;
};

static void DeleteVertexArraysOES(PP_Resource context, GLsizei n, const GLuint* arrays)
{
    LOG_NP;
};

static GLboolean IsVertexArrayOES(PP_Resource context, GLuint array)
{
    LOG_NP;
    return 0;
};

static void BindVertexArrayOES(PP_Resource context, GLuint array)
{
    LOG_NP;
};

struct PPB_OpenGLES2VertexArrayObject_1_0 PPB_OpenGLES2VertexArrayObject_1_0_instance =
{
    .GenVertexArraysOES = GenVertexArraysOES,
    .DeleteVertexArraysOES = DeleteVertexArraysOES,
    .IsVertexArrayOES = IsVertexArrayOES,
    .BindVertexArrayOES = BindVertexArrayOES,
};
