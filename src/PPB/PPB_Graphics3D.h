#ifndef PPB_Graphics3D_h
#define PPB_Graphics3D_h

#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_size.h>

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

typedef struct graphics_3d_desc
{
    PP_Instance instance_id;
    PP_Resource self;
    PP_Resource share_context;

    EGLDisplay dpy;
    EGLSurface pb;
    EGLContext ctx;
    int num_devices;
    EGLDeviceEXT *devices;
    int nc;
    EGLConfig *configs;
    PFNEGLQUERYDEVICESEXTPROC _eglQueryDevicesEXT;
    PFNEGLQUERYDEVICESTRINGEXTPROC _eglQueryDeviceStringEXT;
    PFNEGLGETPLATFORMDISPLAYEXTPROC _eglGetPlatformDisplayEXT;

} graphics_3d_t;

#endif /*PPB_Graphics2D.h */
