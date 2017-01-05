#ifndef INSTANCE_H
#define INSTANCE_H

#include <limits.h>

#include <ppapi/c/pp_var.h>
#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_resource.h>

typedef struct instance_desc
{
    struct
    {
        char DocumentURL[PATH_MAX];
        char PluginInstanceURL[PATH_MAX];
        char Local[PATH_MAX];
    } paths;
    int width, height;
    PP_Resource message_loop_id;
    PP_Instance instance_id;
    struct PP_Var private_instance_object;
    int is_full_frame, is_full_screen;
    PP_Resource graphics_id;
} instance_t;

#endif /* INSTANCE_H */
