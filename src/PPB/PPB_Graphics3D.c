#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_graphics_3d.h>

#include "log.h"

#include "res.h"

#include "PPB_Graphics3D.h"
#include "PPB_MessageLoop.h"

#include "drvapi_error_string.h"

struct PPB_Graphics3D_1_0 PPB_Graphics3D_1_0_instance;

static void Destructor(graphics_3d_t* graphics_3d)
{
    CUresult e;

    LOG("{%d}", graphics_3d->self);

    if(CUDA_SUCCESS != (e = cuCtxPushCurrent(graphics_3d->cu_ctx)))
        LOG("cuCtxPushCurrent failed: %s", getCudaDrvErrorString(e));

    if(CUDA_SUCCESS != (e = cudaGraphicsUnregisterResource(graphics_3d->pbo_res)))
        LOG("cudaGraphicsUnregisterResource failed: %s", getCudaDrvErrorString(e));

    glDeleteBuffers(1, &graphics_3d->pbo);

    if(CUDA_SUCCESS != (e = cuCtxDestroy(graphics_3d->cu_ctx)))
        LOG("cuCtxDestroy failed: %s", getCudaDrvErrorString(e));


    if(graphics_3d->ctx && graphics_3d->dpy)
        eglDestroyContext(graphics_3d->dpy, graphics_3d->ctx);

    if(graphics_3d->pb && graphics_3d->dpy)
        eglDestroySurface(graphics_3d->dpy, graphics_3d->pb);

    if(graphics_3d->dpy)
        eglTerminate(graphics_3d->dpy);

    if(graphics_3d->configs)
        free(graphics_3d->configs);

    if(graphics_3d->devices)
        free(graphics_3d->devices);
};


/**
 * GetAttribMaxValue() retrieves the maximum supported value for the
 * given attribute. This function may be used to check if a particular
 * attribute value is supported before attempting to create a context.
 *
 * @param[in] instance The module instance.
 * @param[in] attribute The attribute for which maximum value is queried.
 * Attributes that can be queried for include:
 * - <code>PP_GRAPHICS3DATTRIB_ALPHA_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_BLUE_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_GREEN_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_RED_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_DEPTH_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_STENCIL_SIZE</code>
 * - <code>PP_GRAPHICS3DATTRIB_SAMPLES</code>
 * - <code>PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS</code>
 * - <code>PP_GRAPHICS3DATTRIB_WIDTH</code>
 * - <code>PP_GRAPHICS3DATTRIB_HEIGHT</code>
 * @param[out] value The maximum supported value for <code>attribute</code>
 *
 * @return Returns <code>PP_TRUE</code> on success or the following on error:
 * - <code>PP_ERROR_BADRESOURCE</code> if <code>instance</code> is invalid
 * - <code>PP_ERROR_BADARGUMENT</code> if <code>attribute</code> is invalid
 * or <code>value</code> is 0
 */
static int32_t GetAttribMaxValue(PP_Resource instance, int32_t attribute, int32_t* value)
{
    LOG_NP;
    return 0;
};

/**
 * Create() creates and initializes a 3D rendering context.
 * The returned context is off-screen to start with. It must be attached to
 * a plugin instance using <code>PPB_Instance::BindGraphics</code> to draw
 * on the web page.
 *
 * @param[in] instance The module instance.
 *
 * @param[in] share_context The 3D context with which the created context
 * would share resources. If <code>share_context</code> is not 0, then all
 * shareable data, as defined by the client API (note that for OpenGL and
 * OpenGL ES, shareable data excludes texture objects named 0) will be shared
 * by <code>share_context<code>, all other contexts <code>share_context</code>
 * already shares with, and the newly created context. An arbitrary number of
 * <code>PPB_Graphics3D</code> can share data in this fashion.
 *
 * @param[in] attrib_list specifies a list of attributes for the context.
 * It is a list of attribute name-value pairs in which each attribute is
 * immediately followed by the corresponding desired value. The list is
 * terminated with <code>PP_GRAPHICS3DATTRIB_NONE</code>.
 * The <code>attrib_list<code> may be 0 or empty (first attribute is
 * <code>PP_GRAPHICS3DATTRIB_NONE</code>). If an attribute is not
 * specified in <code>attrib_list</code>, then the default value is used
 * (it is said to be specified implicitly).
 * Attributes for the context are chosen according to an attribute-specific
 * criteria. Attributes can be classified into two categories:
 * - AtLeast: The attribute value in the returned context meets or exceeds
 *the value specified in <code>attrib_list</code>.
 * - Exact: The attribute value in the returned context is equal to
 *the value specified in <code>attrib_list</code>.
 *
 * Attributes that can be specified in <code>attrib_list</code> include:
 * - <code>PP_GRAPHICS3DATTRIB_ALPHA_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_BLUE_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_GREEN_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_RED_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_DEPTH_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_STENCIL_SIZE</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_SAMPLES</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS</code>:
 * Category: AtLeast Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_WIDTH</code>:
 * Category: Exact Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_HEIGHT</code>:
 * Category: Exact Default: 0.
 * - <code>PP_GRAPHICS3DATTRIB_SWAP_BEHAVIOR</code>:
 * Category: Exact Default: Implementation defined.
 *
 * @return A <code>PP_Resource</code> containing the 3D graphics context if
 * successful or 0 if unsuccessful.
 */
static PP_Resource Create(PP_Instance instance, PP_Resource share_context, const int32_t attrib_list[])
{
    CUresult e;
    CUcontext cu_ctx_pop;
    int i, major, minor, num_devices = 0;
    int res = res_create(sizeof(graphics_3d_t), &PPB_Graphics3D_1_0_instance, (res_destructor_t)Destructor);

    int attribs[] =
    {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
        EGL_NONE
    };

    int pbattribs[] =
    {
        EGL_WIDTH, 1920,
        EGL_HEIGHT, 1080,
        EGL_NONE
    };

    for(i = 0; attrib_list[i] != PP_GRAPHICS3DATTRIB_NONE; i++)
    {
        /* rebuild attr list */
        switch(attrib_list[i])
        {
            case PP_GRAPHICS3DATTRIB_ALPHA_SIZE:
                LOG("ALPHA_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_BLUE_SIZE:
                LOG("BLUE_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_GREEN_SIZE:
                LOG("GREEN_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_RED_SIZE:
                LOG("RED_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_DEPTH_SIZE:
                LOG("DEPTH_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_STENCIL_SIZE:
                LOG("STENCIL_SIZE=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_SAMPLES:
                LOG("SAMPLES=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS:
                LOG("SAMPLE_BUFFERS=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_WIDTH:
                LOG("WIDTH=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_HEIGHT:
                LOG("HEIGHT=%d", (int)attrib_list[i + 1]); i++;
                break;

            case PP_GRAPHICS3DATTRIB_SWAP_BEHAVIOR:
                LOG("SWAP_BEHAVIOR=0x%.8X (%s)", (int)attrib_list[i + 1],
                    attrib_list[i + 1] == PP_GRAPHICS3DATTRIB_BUFFER_PRESERVED ? "PRESERVED" :
                    attrib_list[i + 1] == PP_GRAPHICS3DATTRIB_BUFFER_DESTROYED ? "DESTROYED" :
                    "UNKNOWN"); i++;
                break;

            case PP_GRAPHICS3DATTRIB_GPU_PREFERENCE:
                LOG("PP_GRAPHICS3DATTRIB_GPU_PREFERENCE=0x%.8X (%s)", attrib_list[i + 1],
                    attrib_list[i + 1] == PP_GRAPHICS3DATTRIB_GPU_PREFERENCE_LOW_POWER ? "LOW_POWER" :
                    attrib_list[i + 1] == PP_GRAPHICS3DATTRIB_GPU_PREFERENCE_PERFORMANCE ? "PERFORMANCE" :
                    "UNKNOWN"); i++;
                break;

            default:
                LOG("Unknow attr 0x%.8X", (int)attrib_list[i]);
                break;
        };
    };

    graphics_3d_t* graphics_3d = (graphics_3d_t*)res_private(res);

    graphics_3d->instance_id = instance;
    graphics_3d->self = res;
    graphics_3d->share_context = share_context;

    LOG("res=%d share_context=%d", res, share_context);

    // load function
    if(!(graphics_3d->_eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT")))
    {
        LOG("eglGetProcAddress(\"eglQueryDevicesEXT\") == NULL");
        res_release(res);
        return 0;
    };
    if(!(graphics_3d->_eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT")))
    {
        LOG("eglGetProcAddress(\"eglQueryDeviceStringEXT\") == NULL");
        res_release(res);
        return 0;
    };
    if(!(graphics_3d->_eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT")))
    {
        LOG("eglGetProcAddress(\"eglGetPlatformDisplayEXT\") == NULL");
        res_release(res);
        return 0;
    };

    if(!graphics_3d->_eglQueryDevicesEXT(0, NULL, &graphics_3d->num_devices) || graphics_3d->num_devices < 1)
    {
        LOG("_eglQueryDevicesEXT: num_devices=%d", graphics_3d->num_devices);
        res_release(res);
        return 0;
    };

    graphics_3d->devices = (EGLDeviceEXT*)malloc(sizeof(EGLDeviceEXT) * graphics_3d->num_devices);

    if(!graphics_3d->_eglQueryDevicesEXT(graphics_3d->num_devices, graphics_3d->devices, &graphics_3d->num_devices) || graphics_3d->num_devices < 1)
    {
        LOG("_eglQueryDevicesEXT: num_devices=%d", num_devices);
        res_release(res);
        return 0;
    };

    for(i = 0; i < graphics_3d->num_devices; i++)
    {
        const char *devstr = graphics_3d->_eglQueryDeviceStringEXT(graphics_3d->devices[i], EGL_DRM_DEVICE_FILE_EXT);

        LOG("Device 0x%p: %s", graphics_3d->devices[i], devstr ? devstr : "NULL");
    }

    if((graphics_3d->dpy = graphics_3d->_eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, graphics_3d->devices[0], NULL)) == NULL)
    {
        LOG("_eglQueryDevicesEXT: num_devices=%d", graphics_3d->num_devices);
        res_release(res);
        return 0;
    };

    if(!eglInitialize(graphics_3d->dpy, &major, &minor))
    {
        LOG("eglInitialize failed");
        res_release(res);
        return 0;
    };

    LOG("EGL version %d.%d", major, minor);

    if(!eglChooseConfig(graphics_3d->dpy, attribs, NULL, 0, &graphics_3d->nc) || graphics_3d->nc < 1)
    {
        LOG("eglChooseConfig failed");
        res_release(res);
        return 0;
    };

    graphics_3d->configs = (EGLConfig *)malloc(sizeof(EGLConfig) * graphics_3d->nc);

    if(!eglChooseConfig(graphics_3d->dpy, attribs, graphics_3d->configs, graphics_3d->nc, &graphics_3d->nc) || graphics_3d->nc < 1)
    {
        LOG("eglChooseConfig failed");
        res_release(res);
        return 0;
    };

    if((graphics_3d->pb = eglCreatePbufferSurface(graphics_3d->dpy, graphics_3d->configs[0], pbattribs)) == NULL)
    {
        LOG("eglCreatePbufferSurface failed");
        res_release(res);
        return 0;
    };

    if (!eglBindAPI(EGL_OPENGL_API))
    {
        LOG("eglBindAPI failed");
        res_release(res);
        return 0;
    };


    if((graphics_3d->ctx = eglCreateContext(graphics_3d->dpy, graphics_3d->configs[0], NULL, NULL)) == NULL)
    {
        LOG("eglCreateContext failed");
        res_release(res);
        return 0;
    };

    if(!eglMakeCurrent(graphics_3d->dpy, graphics_3d->pb, graphics_3d->pb, graphics_3d->ctx))
    {
        LOG("eglCreateContext failed");
        res_release(res);
        return 0;
    };

    if(CUDA_SUCCESS != (e = cuInit(0)))
    {
        LOG("cuInit failed");
        res_release(res);
        return 0;
    };

    if(CUDA_SUCCESS != (e = cuCtxCreate(&graphics_3d->cu_ctx, CU_CTX_BLOCKING_SYNC, graphics_3d->cu_dev)))
    {
        LOG("cuCtxCreate failed: %s", getCudaDrvErrorString(e));
        res_release(res);
        return 0;
    };

    glGenBuffers(1, &graphics_3d->pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, graphics_3d->pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 1920 * 1080 * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    if(CUDA_SUCCESS != (e = cudaGraphicsGLRegisterBuffer
        (&graphics_3d->pbo_res, graphics_3d->pbo, cudaGraphicsMapFlagsWriteDiscard)))
    {
        LOG("cudaGraphicsGLRegisterBuffer failed: %s", getCudaDrvErrorString(e));
        res_release(res);
        return 0;
    };


//    LOG("graphics_3d->pb=%d", graphics_3d->pb);

    if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
    {
        LOG("cuCtxCreate failed: %s", getCudaDrvErrorString(e));
        res_release(res);
        return 0;
    };

    return res;
};

/**
 * IsGraphics3D() determines if the given resource is a valid
 * <code>Graphics3D</code> context.
 *
 * @param[in] resource A <code>Graphics3D</code> context resource.
 *
 * @return PP_TRUE if the given resource is a valid <code>Graphics3D</code>,
 * <code>PP_FALSE</code> if it is an invalid resource or is a resource of
 * another type.
 */
static PP_Bool IsGraphics3D(PP_Resource res)
{
    return (res_interface(res) == &PPB_Graphics3D_1_0_instance);
};

/**
 * GetAttribs() retrieves the value for each attribute in
 * <code>attrib_list</code>.
 *
 * @param[in] context The 3D graphics context.
 * @param[in,out] attrib_list The list of attributes that are queried.
 * <code>attrib_list</code> has the same structure as described for
 * <code>PPB_Graphics3D::Create</code>. It is both input and output
 * structure for this function. All attributes specified in
 * <code>PPB_Graphics3D::Create</code> can be queried for.
 *
 * @return Returns <code>PP_OK</code> on success or:
 * - <code>PP_ERROR_BADRESOURCE</code> if context is invalid
 * - <code>PP_ERROR_BADARGUMENT</code> if attrib_list is 0 or any attribute
 * in the <code>attrib_list</code> is not a valid attribute.
 *
 * <strong>Example usage:</strong> To get the values for rgb bits in the
 * color buffer, this function must be called as following:
 * @code
 * int attrib_list[] = {PP_GRAPHICS3DATTRIB_RED_SIZE, 0,
 *PP_GRAPHICS3DATTRIB_GREEN_SIZE, 0,
 *PP_GRAPHICS3DATTRIB_BLUE_SIZE, 0,
 *PP_GRAPHICS3DATTRIB_NONE};
 * GetAttribs(context, attrib_list);
 * int red_bits = attrib_list[1];
 * int green_bits = attrib_list[3];
 * int blue_bits = attrib_list[5];
 * @endcode
 */
static int32_t GetAttribs(PP_Resource context, int32_t attrib_list[])
{
    LOG_NP;
    return 0;
};

/**
 * SetAttribs() sets the values for each attribute in
 * <code>attrib_list</code>.
 *
 * @param[in] context The 3D graphics context.
 * @param[in] attrib_list The list of attributes whose values need to be set.
 * <code>attrib_list</code> has the same structure as described for
 * <code>PPB_Graphics3D::Create</code>.
 * Attributes that can be specified are:
 * - <code>PP_GRAPHICS3DATTRIB_SWAP_BEHAVIOR</code>
 *
 * @return Returns <code>PP_OK</code> on success or:
 * - <code>PP_ERROR_BADRESOURCE</code> if <code>context</code> is invalid.
 * - <code>PP_ERROR_BADARGUMENT</code> if <code>attrib_list</code> is 0 or
 * any attribute in the <code>attrib_list</code> is not a valid attribute.
 */
static int32_t SetAttribs(PP_Resource context, const int32_t attrib_list[])
{
    LOG_NP;
    return 0;
};

/**
 * GetError() returns the current state of the given 3D context.
 *
 * The recoverable error conditions that have no side effect are
 * detected and returned immediately by all functions in this interface.
 * In addition the implementation may get into a fatal state while
 * processing a command. In this case the application must destroy the
 * context and reinitialize client API state and objects to continue
 * rendering.
 *
 * Note that the same error code is also returned in the SwapBuffers callback.
 * It is recommended to handle error in the SwapBuffers callback because
 * GetError is synchronous. This function may be useful in rare cases where
 * drawing a frame is expensive and you want to verify the result of
 * ResizeBuffers before attempting to draw a frame.
 *
 * @param[in] The 3D graphics context.
 * @return Returns:
 * - <code>PP_OK</code> if no error
 * - <code>PP_ERROR_NOMEMORY</code>
 * - <code>PP_ERROR_CONTEXT_LOST</code>
 */
static int32_t GetError(PP_Resource context)
{
    GLenum r;

    LOG_NP;

    r = glGetError();

    switch(r)
    {
        case GL_NO_ERROR: return PP_OK;
        case GL_OUT_OF_MEMORY: return PP_ERROR_NOMEMORY;
        default: return PP_ERROR_CONTEXT_LOST;
    };
};

/**
 * ResizeBuffers() resizes the backing surface for context.
 *
 * If the surface could not be resized due to insufficient resources,
 * <code>PP_ERROR_NOMEMORY</code> error is returned on the next
 * <code>SwapBuffers</code> callback.
 *
 * @param[in] context The 3D graphics context.
 * @param[in] width The width of the backing surface.
 * @param[in] height The height of the backing surface.
 * @return Returns <code>PP_OK</code> on success or:
 * - <code>PP_ERROR_BADRESOURCE</code> if context is invalid.
 * - <code>PP_ERROR_BADARGUMENT</code> if the value specified for
 * <code>width</code> or <code>height</code> is less than zero.
 */
static int32_t ResizeBuffers(PP_Resource context, int32_t width, int32_t height)
{
    LOG_NP;
    return 0;
};

/**
 * SwapBuffers() makes the contents of the color buffer available for
 * compositing. This function has no effect on off-screen surfaces - ones not
 * bound to any plugin instance. The contents of ancillary buffers are always
 * undefined after calling <code>SwapBuffers</code>. The contents of the color
 * buffer are undefined if the value of the
 * <code>PP_GRAPHICS3DATTRIB_SWAP_BEHAVIOR</code> attribute of context is not
 * <code>PP_GRAPHICS3DATTRIB_BUFFER_PRESERVED</code>.
 *
 * <code>SwapBuffers</code> runs in asynchronous mode. Specify a callback
 * function and the argument for that callback function. The callback function
 * will be executed on the calling thread after the color buffer has been
 * composited with rest of the html page. While you are waiting for a
 * SwapBuffers callback, additional calls to SwapBuffers will fail.
 *
 * Because the callback is executed (or thread unblocked) only when the
 * plugin's current state is actually on the screen, this function provides a
 * way to rate limit animations. By waiting until the image is on the screen
 * before painting the next frame, you can ensure you're not generating
 * updates faster than the screen can be updated.
 *
 * SwapBuffers performs an implicit flush operation on context.
 * If the context gets into an unrecoverable error condition while
 * processing a command, the error code will be returned as the argument
 * for the callback. The callback may return the following error codes:
 * - <code>PP_ERROR_NOMEMORY</code>
 * - <code>PP_ERROR_CONTEXT_LOST</code>
 * Note that the same error code may also be obtained by calling GetError.
 *
 * @param[in] context The 3D graphics context.
 * @param[in] callback The callback that will executed when
 * <code>SwapBuffers</code> completes.
 *
 * @return Returns PP_OK on success or:
 * - <code>PP_ERROR_BADRESOURCE</code> if context is invalid.
 * - <code>PP_ERROR_BADARGUMENT</code> if callback is invalid.
 *
 */

static int swaps = 0;
static int32_t SwapBuffers(PP_Resource context, struct PP_CompletionCallback callback)
{
    void * devPtr;
    size_t size;

    CUresult e;
    CUcontext cu_ctx_pop;

    graphics_3d_t* graphics_3d = (graphics_3d_t*)res_private(context);
    int gWidth = 1920, gHeight = 1080;

#if 0
//    graphics_3d_t* graphics_3d = (graphics_3d_t*)res_private(context);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, gWidth, gHeight, GL_RGBA, GL_UNSIGNED_BYTE, image);

#endif

    // copy current buffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER, graphics_3d->pbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, gWidth, gHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    if(CUDA_SUCCESS != (e = cuCtxPushCurrent(graphics_3d->cu_ctx)))
        LOG("cuCtxCreate failed: %s", getCudaDrvErrorString(e));
    if(CUDA_SUCCESS != (e = cudaGraphicsMapResources(1, &graphics_3d->pbo_res, 0)))
        LOG("cudaGraphicsMapResources failed: %s", getCudaDrvErrorString(e));
    if(CUDA_SUCCESS != (e = cudaGraphicsResourceGetMappedPointer(&devPtr, &size, graphics_3d->pbo_res)))
    {
        LOG("cudaGraphicsResourceGetMappedPointer failed: %s", getCudaDrvErrorString(e));

    }
    else
    {
        FILE *f;
        char path[PATH_MAX];

        LOG("cudaGraphicsResourceGetMappedPointer: devPtr=%p, size=%ld", devPtr, size);

        GLubyte *image = calloc(1, size);

        cudaMemcpy(image, devPtr, size, cudaMemcpyDeviceToHost);

        snprintf(path, sizeof(path), "/tmp/tracer-%.6d-%.6d.bin", getpid(), swaps++);
#if 0
        f = fopen(path, "wb");
        fwrite(image, 1, size, f);
        fclose(f);
#endif

        LOG("saved [%s]", path);

        free(image);
    };
    if(CUDA_SUCCESS != (e = cudaGraphicsUnmapResources(1, &graphics_3d->pbo_res, 0)))
        LOG("cudaGraphicsUnmapResources failed: %s", getCudaDrvErrorString(e));
    if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
        LOG("cuCtxCreate failed: %s", getCudaDrvErrorString(e));

    PPB_MessageLoop_push(0, callback, 0, PP_OK);

    LOG("");

    return PP_OK;
};

/**
 * <code>PPB_Graphics3D</code> defines the interface for a 3D graphics context.
 * <strong>Example usage from plugin code:</strong>
 *
 * <strong>Setup:</strong>
 * @code
 * PP_Resource context;
 * int32_t attribs[] = {PP_GRAPHICS3DATTRIB_WIDTH, 800,
 *                      PP_GRAPHICS3DATTRIB_HEIGHT, 800,
 *                      PP_GRAPHICS3DATTRIB_NONE};
 * context = g3d->Create(instance, 0, attribs);
 * inst->BindGraphics(instance, context);
 * @endcode
 *
 * <strong>Present one frame:</strong>
 * @code
 * PP_CompletionCallback callback = {
 *   DidFinishSwappingBuffers, 0, PP_COMPLETIONCALLBACK_FLAG_NONE,
 * };
 * gles2->Clear(context, GL_COLOR_BUFFER_BIT);
 * g3d->SwapBuffers(context, callback);
 * @endcode
 *
 * <strong>Shutdown:</strong>
 * @code
 * core->ReleaseResource(context);
 * @endcode
 */
struct PPB_Graphics3D_1_0 PPB_Graphics3D_1_0_instance =
{
    .GetAttribMaxValue = GetAttribMaxValue,
    .Create = Create,
    .IsGraphics3D = IsGraphics3D,
    .GetAttribs = GetAttribs,
    .SetAttribs = SetAttribs,
    .GetError = GetError,
    .ResizeBuffers = ResizeBuffers,
    .SwapBuffers = SwapBuffers,
};
