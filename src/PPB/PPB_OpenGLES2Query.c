#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void GenQueriesEXT(PP_Resource context, GLsizei n, GLuint* queries)
{
    LOG_NP;
};

static void DeleteQueriesEXT(PP_Resource context,
                           GLsizei n,
                           const GLuint* queries)
{
    LOG_NP;
};

static GLboolean IsQueryEXT(PP_Resource context, GLuint id)
{
    LOG_NP;
    return 0;
};

static void BeginQueryEXT(PP_Resource context, GLenum target, GLuint id)
{
    LOG_NP;
};

static void EndQueryEXT(PP_Resource context, GLenum target)
{
    LOG_NP;
};

static void GetQueryivEXT(PP_Resource context,
                        GLenum target,
                        GLenum pname,
                        GLint* params)
{
    LOG_NP;
};

static void GetQueryObjectuivEXT(PP_Resource context,
                               GLuint id,
                               GLenum pname,
                               GLuint* params)
{
    LOG_NP;
};


struct PPB_OpenGLES2Query_1_0 PPB_OpenGLES2Query_1_0_instance =
{
    .GenQueriesEXT = GenQueriesEXT,
    .DeleteQueriesEXT = DeleteQueriesEXT,
    .IsQueryEXT = IsQueryEXT,
    .BeginQueryEXT = BeginQueryEXT,
    .EndQueryEXT = EndQueryEXT,
    .GetQueryivEXT = GetQueryivEXT,
    .GetQueryObjectuivEXT = GetQueryObjectuivEXT,
};
