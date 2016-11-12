#ifndef PPB_URLUtil_Dev_h
#define PPB_URLUtil_Dev_h

#include <ppapi/c/pp_var.h>
#include <ppapi/c/dev/ppb_url_util_dev.h>

void uriparser_parse_var(struct PP_Var url, struct PP_URLComponents_Dev* comp);
void uriparser_parse(const char* url, struct PP_URLComponents_Dev* comp);

#endif /* PPB_URLUtil_Dev_h */
