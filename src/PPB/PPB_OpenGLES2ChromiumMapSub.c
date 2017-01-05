#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_opengles2.h>

#include "log.h"

#include "PPB_Graphics3D.h"

typedef struct MapBufferSubDataCHROMIUM_desc
{
    GLuint target;
    GLintptr offset;
    GLsizeiptr size;
    GLenum access;
} MapBufferSubDataCHROMIUM_t;

typedef struct MapTexSubImage2DCHROMIUM_desc
{
    GLenum target;
    GLint level;
    GLint xoffset;
    GLint yoffset;
    GLsizei width;
    GLsizei height;
    GLenum format;
    GLenum type;
    GLenum access;
} MapTexSubImage2DCHROMIUM_t;


// https://github.com/adobe/chromium/blob/master/gpu/GLES2/extensions/CHROMIUM/CHROMIUM_map_sub.txt

static void* MapBufferSubDataCHROMIUM(PP_Resource context,
                                    GLuint target,
                                    GLintptr offset,
                                    GLsizeiptr size,
                                    GLenum access)
{
    MapBufferSubDataCHROMIUM_t *d;

    LOG_TD;

    d = (MapBufferSubDataCHROMIUM_t*)malloc(sizeof(MapBufferSubDataCHROMIUM_t) + size);

    d->target = target;
    d->offset = offset;
    d->size = size;
    d->access = access;

    return (unsigned char*)d + sizeof(MapBufferSubDataCHROMIUM_t);
};

static void UnmapBufferSubDataCHROMIUM(PP_Resource context, const void* mem)
{
    MapBufferSubDataCHROMIUM_t *d = (MapBufferSubDataCHROMIUM_t *)((unsigned char*)mem - sizeof(MapBufferSubDataCHROMIUM_t));

    LOG_TD;

    glBufferSubData(d->target, d->offset, d->size, mem);

    free(d);
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
    int size;
    MapTexSubImage2DCHROMIUM_t *d;

    LOG_TD;

    size = width * height * 8;

    d = (MapTexSubImage2DCHROMIUM_t*)malloc(sizeof(MapTexSubImage2DCHROMIUM_t) + size);

    d->target = target;
    d->level = level;
    d->xoffset = xoffset;
    d->yoffset = yoffset;
    d->width = width;
    d->height = height;
    d->format = format;
    d->type = type;
    d->access = access;

    LOG_D("xoffset=%d, yoffset=%d, width=%d, height=%d", d->xoffset, d->yoffset, d->width, d->height);

    return (unsigned char*)d + sizeof(MapTexSubImage2DCHROMIUM_t);
};

static void UnmapTexSubImage2DCHROMIUM(PP_Resource context, const void* mem)
{
    MapTexSubImage2DCHROMIUM_t *d = (MapTexSubImage2DCHROMIUM_t *)((unsigned char*)mem - sizeof(MapTexSubImage2DCHROMIUM_t));

    LOG_TD;

    LOG_D("xoffset=%d, yoffset=%d, width=%d, height=%d", d->xoffset, d->yoffset, d->width, d->height);

    glTexSubImage2D
    (
        d->target,
        d->level,
        d->xoffset,
        d->yoffset,
        d->width,
        d->height,
        d->format,
        d->type,
        mem
    );

    free(d);
};

struct PPB_OpenGLES2ChromiumMapSub_1_0 PPB_OpenGLES2ChromiumMapSub_1_0_instance =
{
    .MapBufferSubDataCHROMIUM = MapBufferSubDataCHROMIUM,
    .UnmapBufferSubDataCHROMIUM = UnmapBufferSubDataCHROMIUM,
    .MapTexSubImage2DCHROMIUM = MapTexSubImage2DCHROMIUM,
    .UnmapTexSubImage2DCHROMIUM = UnmapTexSubImage2DCHROMIUM,
};
