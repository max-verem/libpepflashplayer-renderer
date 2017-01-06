#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash_fullscreen.h>

#include "log.h"
#include "instance.h"
#include "res.h"

/**
 * IsFullscreen() checks whether the module instance is currently in
 * fullscreen mode.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 *
 * @return <code>PP_TRUE</code> if the module instance is in fullscreen mode,
 * <code>PP_FALSE</code> if the module instance is not in fullscreen mode.
 */
static PP_Bool IsFullscreen(PP_Instance instance)
{
    instance_t* inst = (instance_t*)res_private(instance);
    LOG_N("inst->is_full_screen=%d", inst->is_full_screen);
    return inst->is_full_screen;
};

/**
 * SetFullscreen() switches the module instance to and from fullscreen
 * mode.
 *
 * The transition to and from fullscreen mode is asynchronous. During the
 * transition, IsFullscreen() will return the previous value and
 * no 2D or 3D device can be bound. The transition ends at DidChangeView()
 * when IsFullscreen() returns the new value. You might receive other
 * DidChangeView() calls while in transition.
 *
 * The transition to fullscreen mode can only occur while the browser is
 * processing a user gesture, even if <code>PP_TRUE</code> is returned.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 * @param[in] fullscreen <code>PP_TRUE</code> to enter fullscreen mode, or
 * <code>PP_FALSE</code> to exit fullscreen mode.
 *
 * @return <code>PP_TRUE</code> on success or <code>PP_FALSE</code> on
 * failure.
 */
static PP_Bool SetFullscreen(PP_Instance instance, PP_Bool fullscreen)
{
    instance_t* inst = (instance_t*)res_private(instance);
    LOG_N("want fullscreen=%d, origin fullscreen=%d", fullscreen, inst->is_full_screen);
    return 1;
};

/**
 * GetScreenSize() gets the size of the screen in pixels. The module instance
 * will be resized to this size when SetFullscreen() is called to enter
 * fullscreen mode.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 * @param[out] size The size of the entire screen in pixels.
 *
 * @return <code>PP_TRUE</code> on success or <code>PP_FALSE</code> on
 * failure.
 */
static PP_Bool GetScreenSize(PP_Instance instance, struct PP_Size* size)
{
    instance_t* inst = (instance_t*)res_private(instance);
    size->width = inst->width;
    size->height = inst->height;
    LOG_N("size->width=%d, size->height=%d", size->width, size->height);
    return PP_TRUE;
};

struct PPB_FlashFullscreen_1_0 PB_FlashFullscreen_1_0_instance =
{
    .IsFullscreen = IsFullscreen,
    .SetFullscreen = SetFullscreen,
    .GetScreenSize = GetScreenSize,
};
