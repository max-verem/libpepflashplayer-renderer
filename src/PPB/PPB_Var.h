#ifndef PPB_Var_h
#define PPB_Var_h

#include <ppapi/c/pp_var.h>
#include <string.h>

struct PP_Var VarFromUtf8(const char* data, uint32_t len);
void PPB_Var_AddRef(struct PP_Var var);
void PPB_Var_Release(struct PP_Var var);
const char* VarToUtf8(struct PP_Var var, uint32_t* len);
struct PP_Var VarFromResource(PP_Resource resource);
void PPB_Var_Dump(const char* name, struct PP_Var var);

#define VarFromUtf8_c(CH) VarFromUtf8(CH, strlen(CH))

#endif /* PPB_Var_h */
