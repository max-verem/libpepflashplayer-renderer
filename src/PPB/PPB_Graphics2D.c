#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_graphics_2d.h>

#include "log.h"
#include "res.h"

#include "PPB_Graphics2D.h"

struct PPB_Graphics2D_1_1 PPB_Graphics2D_1_1_instance;

static void Destructor(graphics_2d_t* url_req)
{
};

static PP_Resource Create(PP_Instance instance,
    const struct PP_Size* size, PP_Bool is_always_opaque)
{
    int res = res_create(sizeof(graphics_2d_t), &PPB_Graphics2D_1_1_instance, (res_destructor_t)Destructor);

    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    graphics_2d->instance_id = instance;
    graphics_2d->size = *size;
    graphics_2d->is_always_opaque = is_always_opaque;
    graphics_2d->scale = 1.0;

    LOG("res=%d, size->width=%d, size->height=%d", res, size->width, size->height);

    return res;
};

static PP_Bool IsGraphics2D(PP_Resource res)
{
    return (res_interface(res) == &PPB_Graphics2D_1_1_instance);
};

static PP_Bool Describe(PP_Resource res, struct PP_Size* size, PP_Bool* is_always_opaque)
{
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    *size = graphics_2d->size;
    *is_always_opaque = graphics_2d->is_always_opaque;

    return 1;
};

static void PaintImageData(PP_Resource graphics_2d,
    PP_Resource image_data, const struct PP_Point* top_left, const struct PP_Rect* src_rect)
{
    LOG_NP;
};

static void Scroll(PP_Resource graphics_2d,
    const struct PP_Rect* clip_rect, const struct PP_Point* amount)
{
    LOG_NP;
};

static void ReplaceContents(PP_Resource graphics_2d, PP_Resource image_data)
{
    LOG_NP;
};

static int32_t Flush(PP_Resource graphics_2d, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static PP_Bool SetScale(PP_Resource res, float scale)
{
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    graphics_2d->scale = scale;

    return 1;
};

static float GetScale(PP_Resource res)
{
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    return graphics_2d->scale;
};

struct PPB_Graphics2D_1_1 PPB_Graphics2D_1_1_instance =
{
    .Create = Create,
    .IsGraphics2D = IsGraphics2D,
    .Describe = Describe,
    .PaintImageData = PaintImageData,
    .Scroll = Scroll,
    .ReplaceContents = ReplaceContents,
    .Flush = Flush,
    .SetScale = SetScale,
    .GetScale = GetScale,
};
