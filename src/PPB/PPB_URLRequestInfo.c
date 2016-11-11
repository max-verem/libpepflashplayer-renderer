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
#include "impl/PPB_Var.h"

#define MAX_PROPS (PP_URLREQUESTPROPERTY_CUSTOMUSERAGENT + 1)

typedef struct url_request_info_desc
{
    PP_Instance instance_id;
    struct PP_Var props[MAX_PROPS];
} url_request_info_t;

static void Destructor(url_request_info_t* url_req)
{
    int i;

    for(i = 0; i < MAX_PROPS; i++)
        if(url_req->props[i].type != PP_VARTYPE_UNDEFINED)
            PPB_Var_Release(url_req->props[i]);
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

    LOG("property=%d", property);

    return 0;
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
    LOG_NP;
    return 0;
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
    LOG_NP;
    return 0;
};

struct PPB_URLRequestInfo_1_0 PPB_URLRequestInfo_1_0_instance =
{
    .Create = Create,
    .IsURLRequestInfo = IsURLRequestInfo,
    .SetProperty = SetProperty,
    .AppendDataToBody = AppendDataToBody,
    .AppendFileToBody = AppendFileToBody,
};
