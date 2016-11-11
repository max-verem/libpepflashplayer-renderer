#ifndef RES_H
#define RES_H

#include <ppapi/c/pp_resource.h>

int res_begin();
int res_end();

typedef void (*res_destructor_t)(void* priv);

PP_Resource res_create(size_t private_size, void* interface, res_destructor_t destructor);
int res_add_ref(PP_Resource res);
int res_release(PP_Resource res);
void* res_private(PP_Resource res);
void* res_interface(PP_Resource res);
int res_private_size(PP_Resource res);

#endif /* RES_H */
