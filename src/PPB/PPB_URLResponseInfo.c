#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_url_response_info.h>

#include "log.h"

/**
 * IsURLResponseInfo() determines if a response is a
 * <code>URLResponseInfo</code>.
 *
 * @param[in] resource A <code>PP_Resource</code> corresponding to a
 * <code>URLResponseInfo</code>.
 *
 * @return <code>PP_TRUE</code> if the resource is a
 * <code>URLResponseInfo</code>, <code>PP_FALSE</code> if the resource is
 * invalid or some type other than <code>URLResponseInfo</code>.
 */
static PP_Bool IsURLResponseInfo(PP_Resource resource)
{
    LOG_NP;
    return 0;
};


/**
 * GetProperty() gets a response property.
 *
 * @param[in] request A <code>PP_Resource</code> corresponding to a
 * <code>URLResponseInfo</code>.
 * @param[in] property A <code>PP_URLResponseProperty</code> identifying
 * the type of property in the response.
 *
 * @return A <code>PP_Var</code> containing the response property value if
 * successful, <code>PP_VARTYPE_VOID</code> if an input parameter is invalid.
 */
static struct PP_Var GetProperty(PP_Resource response, PP_URLResponseProperty property)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


/**
 * GetBodyAsFileRef() returns a FileRef pointing to the file containing the
 * response body.This is only valid if
 * <code>PP_URLREQUESTPROPERTY_STREAMTOFILE</code> was set on the
 * <code>URLRequestInfo</code> used to produce this response.This file
 * remains valid until the <code>URLLoader</code> associated with this
 * <code>URLResponseInfo</code> is closed or destroyed.
 *
 * @param[in] request A <code>PP_Resource</code> corresponding to a
 * <code>URLResponseInfo</code>.
 *
 * @return A <code>PP_Resource</code> corresponding to a <code>FileRef</code>
 * if successful, 0 if <code>PP_URLREQUESTPROPERTY_STREAMTOFILE</code> was
 * not requested or if the <code>URLLoader</code> has not been opened yet.
 */
static PP_Resource GetBodyAsFileRef(PP_Resource response)
{
    LOG_NP;
    return 0;
};

struct PPB_URLResponseInfo_1_0 PPB_URLResponseInfo_1_0_instance =
{
    .IsURLResponseInfo = IsURLResponseInfo,
    .GetProperty = GetProperty,
    .GetBodyAsFileRef = GetBodyAsFileRef,
};
