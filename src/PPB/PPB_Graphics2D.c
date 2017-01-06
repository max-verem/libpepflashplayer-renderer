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

static void Destructor(graphics_2d_t* ctx)
{
    LOG_D("{%d}", ctx->self);
};

static PP_Resource Create(PP_Instance instance,
    const struct PP_Size* size, PP_Bool is_always_opaque)
{
    int res = res_create(sizeof(graphics_2d_t), &PPB_Graphics2D_1_1_instance, (res_destructor_t)Destructor);
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    graphics_2d->instance_id = instance;
    graphics_2d->self = res;
    graphics_2d->size = *size;
    graphics_2d->is_always_opaque = is_always_opaque;
    graphics_2d->scale = 1.0;

    LOG_N("res=%d, size->width=%d, size->height=%d, is_always_opaque=%d",
        res, size->width, size->height, is_always_opaque);

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

/**
 * ReplaceContents() provides a slightly more efficient way to paint the
 * entire module's image. Normally, calling PaintImageData() requires that
 * the browser copy the pixels out of the image and into the graphics
 * context's backing store. This function replaces the graphics context's
 * backing store with the given image, avoiding the copy.
 *
 * The new image must be the exact same size as this graphics context. If the
 * new image uses a different image format than the browser's native bitmap
 * format (use <code>PPB_ImageData.GetNativeImageDataFormat()</code> to
 * retrieve the format), then a conversion will be done inside the browser
 * which may slow the performance a little bit.
 *
 * <strong>Note:</strong> The new image will not be painted until you call
 * Flush().
 *
 * After this call, you should take care to release your references to the
 * image. If you paint to the image after ReplaceContents(), there is the
 * possibility of significant painting artifacts because the page might use
 * partially-rendered data when copying out of the backing store.
 *
 * In the case of an animation, you will want to allocate a new image for the
 * next frame. It is best if you wait until the flush callback has executed
 * before allocating this bitmap. This gives the browser the option of
 * caching the previous backing store and handing it back to you (assuming
 * the sizes match). In the optimal case, this means no bitmaps are allocated
 * during the animation, and the backing store and "front buffer" (which the
 * plugin is painting into) are just being swapped back and forth.
 *
 * @param[in] graphics_2d The 2D Graphics resource.
 * @param[in] image The <code>ImageData</code> to be painted.
 */
static void ReplaceContents(PP_Resource res, PP_Resource image_data)
{
    LOG_N("{%d} image_data=%d", res, image_data);
};

/**
 * Flush() flushes any enqueued paint, scroll, and replace commands to the
 * backing store. This function actually executes the updates, and causes a
 * repaint of the webpage, assuming this graphics context is bound to a module
 * instance.
 *
 * Flush() runs in asynchronous mode. Specify a callback function and the
 * argument for that callback function. The callback function will be
 * executed on the calling thread when the image has been painted to the
 * screen. While you are waiting for a flush callback, additional calls to
 * Flush() will fail.
 *
 * Because the callback is executed (or thread unblocked) only when the
 * instance's image is actually on the screen, this function provides
 * a way to rate limit animations. By waiting until the image is on the
 * screen before painting the next frame, you can ensure you're not
 * flushing 2D graphics faster than the screen can be updated.
 *
 * <strong>Unbound contexts</strong>
 * If the context is not bound to a module instance, you will
 * still get a callback. The callback will execute after Flush() returns
 * to avoid reentrancy. The callback will not wait until anything is
 * painted to the screen because there will be nothing on the screen. The
 * timing of this callback is not guaranteed and may be deprioritized by
 * the browser because it is not affecting the user experience.
 *
 * <strong>Off-screen instances</strong>
 * If the context is bound to an instance that is currently not visible (for
 * example, scrolled out of view) it will behave like the "unbound context"
 * case.
 *
 * <strong>Detaching a context</strong>
 * If you detach a context from a module instance, any pending flush
 * callbacks will be converted into the "unbound context" case.
 *
 * <strong>Released contexts</strong>
 * A callback may or may not get called even if you have released all
 * of your references to the context. This scenario can occur if there are
 * internal references to the context suggesting it has not been internally
 * destroyed (for example, if it is still bound to an instance) or due to
 * other implementation details. As a result, you should be careful to
 * check that flush callbacks are for the context you expect and that
 * you're capable of handling callbacks for unreferenced contexts.
 *
 * <strong>Shutdown</strong>
 * If a module instance is removed when a flush is pending, the
 * callback will not be executed.
 *
 * @param[in] graphics_2d The 2D Graphics resource.
 * @param[in] callback A <code>CompletionCallback</code> to be called when
 * the image has been painted on the screen.
 *
 * @return Returns <code>PP_OK</code> on success or
 * <code>PP_ERROR_BADRESOURCE</code> if the graphics context is invalid,
 * <code>PP_ERROR_BADARGUMENT</code> if the callback is null and flush is
 * being called from the main thread of the module, or
 * <code>PP_ERROR_INPROGRESS</code> if a flush is already pending that has
 * not issued its callback yet.In the failure case, nothing will be updated
 * and no callback will be scheduled.
 */
static int32_t Flush(PP_Resource graphics_2d, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

static PP_Bool SetScale(PP_Resource res, float scale)
{
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    LOG_N("{%d}", res);

    graphics_2d->scale = scale;

    return 1;
};

static float GetScale(PP_Resource res)
{
    graphics_2d_t* graphics_2d = (graphics_2d_t*)res_private(res);

    LOG_N("{%d}", res);

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
