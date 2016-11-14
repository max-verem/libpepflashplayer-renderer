#ifndef PPB_ImageData_h
#define PPB_ImageData_h

#include <ppapi/c/pp_instance.h>
#include <ppapi/c/pp_size.h>

typedef struct image_data_desc
{
    PP_Instance instance_id;

    struct PP_Size size;
    PP_ImageDataFormat format;

    void* map;
    void* buf;
} image_data_t;

#endif /* PPB_ImageData_h */
