#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_var.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_url_request_info.h>

#include "log.h"
#include "res.h"
#include "PPB_Var.h"
#include "PPB_URLRequestInfo.h"

static const char* URLRequestProperties[PP_URLREQUESTPROPERTY_LAST] =
{
    "URL",
    "METHOD",
    "HEADERS",
    "STREAMTOFILE",
    "FOLLOWREDIRECTS",
    "RECORDDOWNLOADPROGRESS",
    "RECORDUPLOADPROGRESS",
    "CUSTOMREFERRERURL",
    "ALLOWCROSSORIGINREQUESTS",
    "ALLOWCREDENTIALS",
    "CUSTOMCONTENTTRANSFERENCODING",
    "PREFETCHBUFFERUPPERTHRESHOLD",
    "PREFETCHBUFFERLOWERTHRESHOLD",
    "CUSTOMUSERAGENT",
};

static void Destructor(url_request_info_t* ctx)
{
    int i;

    LOG("{%d}", ctx->self);

    for(i = 0; i < PP_URLREQUESTPROPERTY_LAST; i++)
        if(ctx->props[i].type != PP_VARTYPE_UNDEFINED)
            PPB_Var_Release(ctx->props[i]);
};

struct PPB_URLRequestInfo_1_0 PPB_URLRequestInfo_1_0_instance;

/**
 * Create() creates a new <code>URLRequestInfo</code> object.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 *
 * @return A <code>PP_Resource</code> identifying the
 * <code>URLRequestInfo</code> if successful, 0 if the instance is invalid.
 */
static PP_Resource Create(PP_Instance instance)
{
    int res = res_create(sizeof(url_request_info_t), &PPB_URLRequestInfo_1_0_instance, (res_destructor_t)Destructor);

    url_request_info_t* url_req = (url_request_info_t*)res_private(res);

    url_req->instance_id = instance;
    url_req->self = res;

    LOG("res=%d", res);

    return res;
};

/**
 * IsURLRequestInfo() determines if a resource is a
 * <code>URLRequestInfo</code>.
 *
 * @param[in] resource A <code>PP_Resource</code> corresponding to a
 * <code>URLRequestInfo</code>.
 *
 * @return <code>PP_TRUE</code> if the resource is a
 * <code>URLRequestInfo</code>, <code>PP_FALSE</code> if the resource is
 * invalid or some type other than <code>URLRequestInfo</code>.
 */
static PP_Bool IsURLRequestInfo(PP_Resource resource)
{
    return (res_interface(resource) == &PPB_URLRequestInfo_1_0_instance);
};

/**
 * SetProperty() sets a request property. The value of the property must be
 * the correct type according to the property being set.
 *
 * @param[in] request A <code>PP_Resource</code> corresponding to a
 * <code>URLRequestInfo</code>.
 * @param[in] property A <code>PP_URLRequestProperty</code> identifying the
 * property to set.
 * @param[in] value A <code>PP_Var</code> containing the property value.
 *
 * @return <code>PP_TRUE</code> if successful, <code>PP_FALSE</code> if any
 * of the parameters are invalid.
 */
static PP_Bool SetProperty(PP_Resource request,
    PP_URLRequestProperty property, struct PP_Var value)
{
    if(!IsURLRequestInfo(request))
        return 0;

    url_request_info_t* url_req = (url_request_info_t*)res_private(request);

    PPB_Var_AddRef(value);

    url_req->props[property] = value;

    LOG("{%d} property=%s", request, URLRequestProperties[property]);
    if(value.type == PP_VARTYPE_STRING)
    {
        LOG("{%d} value=[%s]", request, (char*)res_private(value.value.as_id));
    }
    else if(value.type == PP_VARTYPE_INT32)
    {
        LOG("{%d} value=%d", request, value.value.as_int);
    }
    else if(value.type == PP_VARTYPE_BOOL)
    {
        LOG("{%d} value=%s", request, value.value.as_bool ? "TRUE" : "FALSE");
    };

    return PP_TRUE;
};

/**
 * AppendDataToBody() appends data to the request body. A Content-Length
 * request header will be automatically generated.
 *
 * @param[in] request A <code>PP_Resource</code> corresponding to a
 * <code>URLRequestInfo</code>.
 * @param[in] data A pointer to a buffer holding the data.
 * @param[in] len The length, in bytes, of the data.
 *
 * @return <code>PP_TRUE</code> if successful, <code>PP_FALSE</code> if any
 * of the parameters are invalid.
 *
 *
 */
static PP_Bool AppendDataToBody(PP_Resource request, const void* data, uint32_t len)
{
    url_request_info_t* url_req = (url_request_info_t*)res_private(request);

    LOG("{%d} data=%p, len=%d", request, data, len);

    url_req->DataToBody.data = data;
    url_req->DataToBody.len = len;

    return PP_TRUE;
};


/**
 * AppendFileToBody() appends a file, to be uploaded, to the request body.
 * A content-length request header will be automatically generated.
 *
 * @param[in] request A <code>PP_Resource</code> corresponding to a
 * <code>URLRequestInfo</code>.
 * @param[in] file_ref A <code>PP_Resource</code> corresponding to a file
 * reference.
 * @param[in] start_offset An optional starting point offset within the
 * file.
 * @param[in] number_of_bytes An optional number of bytes of the file to
 * be included. If <code>number_of_bytes</code> is -1, then the sub-range
 * to upload extends to the end of the file.
 * @param[in] expected_last_modified_time An optional (non-zero) last
 * modified time stamp used to validate that the file was not modified since
 * the given time before it was uploaded. The upload will fail with an error
 * code of <code>PP_ERROR_FILECHANGED</code> if the file has been modified
 * since the given time. If <code>expected_last_modified_time</code> is 0,
 * then no validation is performed.
 *
 * @return <code>PP_TRUE</code> if successful, <code>PP_FALSE</code> if any
 * of the parameters are invalid.
 */
static PP_Bool AppendFileToBody(PP_Resource request, PP_Resource file_ref,
    int64_t start_offset, int64_t number_of_bytes, PP_Time expected_last_modified_time)
{
    url_request_info_t* url_req = (url_request_info_t*)res_private(request);

    LOG("{%d}", request);

    res_add_ref(file_ref);

    url_req->FileToBody.file_ref = file_ref;
    url_req->FileToBody.start_offset = start_offset;
    url_req->FileToBody.number_of_bytes = number_of_bytes;
    url_req->FileToBody.expected_last_modified_time = expected_last_modified_time;

    return PP_TRUE;
};

struct PPB_URLRequestInfo_1_0 PPB_URLRequestInfo_1_0_instance =
{
    .Create = Create,
    .IsURLRequestInfo = IsURLRequestInfo,
    .SetProperty = SetProperty,
    .AppendDataToBody = AppendDataToBody,
    .AppendFileToBody = AppendFileToBody,
};
