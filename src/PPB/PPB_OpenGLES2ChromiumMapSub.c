#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

static void* MapBufferSubDataCHROMIUM(PP_Resource context,
                                    GLuint target,
                                    GLintptr offset,
                                    GLsizeiptr size,
                                    GLenum access)
{
    LOG_NP;
    return 0;
};

static void UnmapBufferSubDataCHROMIUM(PP_Resource context, const void* mem)
{
    LOG_NP;
};

static void* MapTexSubImage2DCHROMIUM(PP_Resource context,
                                    GLenum target,
                                    GLint level,
                                    GLint xoffset,
                                    GLint yoffset,
                                    GLsizei width,
                                    GLsizei height,
                                    GLenum format,
                                    GLenum type,
                                    GLenum access)
{
    LOG_NP;
    return 0;
};

static void UnmapTexSubImage2DCHROMIUM(PP_Resource context, const void* mem)
{
    LOG_NP;
};

struct PPB_OpenGLES2ChromiumMapSub_1_0 PPB_OpenGLES2ChromiumMapSub_1_0_instance =
{
    .MapBufferSubDataCHROMIUM = MapBufferSubDataCHROMIUM,
    .UnmapBufferSubDataCHROMIUM = UnmapBufferSubDataCHROMIUM,
    .MapTexSubImage2DCHROMIUM = MapTexSubImage2DCHROMIUM,
    .UnmapTexSubImage2DCHROMIUM = UnmapTexSubImage2DCHROMIUM,
};
