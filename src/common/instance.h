#ifndef INSTANCE_H
#define INSTANCE_H

#include <limits.h>

#include <ppapi/c/pp_var.h>
#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_resource.h>
#include <ppapi/c/dev/ppp_class_deprecated.h>

typedef struct instance_desc
{
    struct
    {
        char DocumentURL[PATH_MAX];
        char PluginInstanceURL[PATH_MAX];
        char Local[PATH_MAX];
    } paths;
    int width, height, fps;
    PP_Resource message_loop_id;
    PP_Instance instance_id;
    struct PP_Var private_instance_object;
    struct PP_Var window_instance_object;
    int is_full_frame, is_full_screen;
    PP_Resource graphics_id;
    struct PP_Var object;
    int (*buffer_swap_begin)(void *app_data, void**, size_t* sz);
    int (*buffer_swap_end)(void *app_data, void**);
    const struct PPP_Class_Deprecated* app_class;
    void* app_data;
} instance_t;

#endif /* INSTANCE_H */
