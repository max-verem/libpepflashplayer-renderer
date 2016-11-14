#ifndef PPB_Graphics2D_h
#define PPB_Graphics2D_h

#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_size.h>

typedef struct graphics_2d_desc
{
    PP_Instance instance_id;

    struct PP_Size size;
    PP_Bool is_always_opaque;
    double scale;
} graphics_2d_t;

#endif /*PPB_Graphics2D.h */
