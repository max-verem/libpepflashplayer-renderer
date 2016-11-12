#ifndef PPB_URLRequestInfo_h
#define PPB_URLRequestInfo_h

#include <ppapi/c/pp_var.h>
#include <ppapi/c/pp_instance.h>
#include <ppapi/c/ppb_url_request_info.h>

#define PP_URLREQUESTPROPERTY_LAST (PP_URLREQUESTPROPERTY_CUSTOMUSERAGENT + 1)

typedef struct url_request_info_desc
{
    PP_Instance instance_id;
    struct PP_Var props[PP_URLREQUESTPROPERTY_LAST];
} url_request_info_t;

#endif /* PPB_URLRequestInfo_h */