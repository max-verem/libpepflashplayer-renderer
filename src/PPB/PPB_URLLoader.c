#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_url_loader.h>

#include "log.h"
#include "res.h"

typedef struct url_loader_desc
{
    PP_Instance instance_id;
} url_loader_t;

static void Destructor(url_loader_t* url_load)
{
};

struct PPB_URLLoader_1_0 PPB_URLLoader_1_0_instance;

/**
 * Create() creates a new <code>URLLoader</code> object. The
 * <code>URLLoader</code> is associated with a particular instance, so that
 * any UI dialogs that need to be shown to the user can be positioned
 * relative to the window containing the instance.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance
 * of a module.
 *
 * @return A <code>PP_Resource</code> corresponding to a URLLoader if
 * successful, 0 if the instance is invalid.
 */
static PP_Resource Create(PP_Instance instance)
{
    int res = res_create(sizeof(url_loader_t), &PPB_URLLoader_1_0_instance, (res_destructor_t)Destructor);

    url_loader_t* url_loader = (url_loader_t*)res_private(res);

    url_loader->instance_id = instance;

    LOG("res=%d", res);

    return res;
};

/**
 * IsURLLoader() determines if a resource is an <code>URLLoader</code>.
 *
 * @param[in] resource A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 *
 * @return <code>PP_TRUE</code> if the resource is a <code>URLLoader</code>,
 * <code>PP_FALSE</code> if the resource is invalid or some type other
 * than <code>URLLoader</code>.
 */
static PP_Bool IsURLLoader(PP_Resource resource)
{
    return (res_interface(resource) == &PPB_URLLoader_1_0_instance);
};


/**
 * Open() begins loading the <code>URLRequestInfo</code>. The operation
 * completes when response headers are received or when an error occurs.Use
 * GetResponseInfo() to access the response headers.
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in] resource A <code>PP_Resource</code> corresponding to a
 * <code>URLRequestInfo</code>.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * asynchronous completion of Open(). This callback will run when response
 * headers for the url are received or error occurred. This callback
 * will only run if Open() returns <code>PP_OK_COMPLETIONPENDING</code>.
 *
 * @return An int32_t containing an error code from <code>pp_errors.h</code>.
 */
static int32_t Open(PP_Resource loader, PP_Resource request_info, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * FollowRedirect() can be invoked to follow a redirect after Open()
 * completed on receiving redirect headers.
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * asynchronous completion of FollowRedirect(). This callback will run when
 * response headers for the redirect url are received or error occurred. This
 * callback will only run if FollowRedirect() returns
 * <code>PP_OK_COMPLETIONPENDING</code>.
 *
 * @return An int32_t containing an error code from <code>pp_errors.h</code>.
 */
static int32_t FollowRedirect(PP_Resource loader, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * GetUploadProgress() returns the current upload progress (which is
 * meaningful after Open() has been called). Progress only refers to the
 * request body and does not include the headers.
 *
 * This data is only available if the <code>URLRequestInfo</code> passed
 * to Open() had the <code>PP_URLREQUESTPROPERTY_REPORTUPLOADPROGRESS</code>
 * property set to PP_TRUE.
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in] bytes_sent The number of bytes sent thus far.
 * @param[in] total_bytes_to_be_sent The total number of bytes to be sent.
 *
 * @return <code>PP_TRUE</code> if the upload progress is available,
 * <code>PP_FALSE</code> if it is not available.
 */
static PP_Bool GetUploadProgress(PP_Resource loader, int64_t* bytes_sent,  int64_t* total_bytes_to_be_sent)
{
    LOG_NP;
    return 0;
};

/**
 * GetDownloadProgress() returns the current download progress, which is
 * meaningful after Open() has been called. Progress only refers to the
 * response body and does not include the headers.
 *
 * This data is only available if the <code>URLRequestInfo</code> passed to
 * Open() had the <code>PP_URLREQUESTPROPERTY_REPORTDOWNLOADPROGRESS</code>
 * property set to <code>PP_TRUE</code>.
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in] bytes_received The number of bytes received thus far.
 * @param[in] total_bytes_to_be_received The total number of bytes to be
 * received. The total bytes to be received may be unknown, in which case
 * <code>total_bytes_to_be_received</code> will be set to -1.
 *
 * @return <code>PP_TRUE</code> if the download progress is available,
 * <code>PP_FALSE</code> if it is not available.
 */
static PP_Bool GetDownloadProgress(PP_Resource loader, int64_t* bytes_received, int64_t* total_bytes_to_be_received)
{
    LOG_NP;
    return 0;
};

/**
 * GetResponseInfo() returns the current <code>URLResponseInfo</code> object.
 *
 * @param[in] instance A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 *
 * @return A <code>PP_Resource</code> corresponding to the
 * <code>URLResponseInfo</code> if successful, 0 if the loader is not a valid
 * resource or if Open() has not been called.
 */
static PP_Resource GetResponseInfo(PP_Resource loader)
{
    LOG_NP;
    return 0;
};


/**
 * ReadResponseBody() is used to read the response body. The size of the
 * buffer must be large enough to hold the specified number of bytes to read.
 * This function might perform a partial read.
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in,out] buffer A pointer to the buffer for the response body.
 * @param[in] bytes_to_read The number of bytes to read.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * asynchronous completion. The callback will run if the bytes (full or
 * partial) are read or an error occurs asynchronously. This callback will
 * run only if this function returns <code>PP_OK_COMPLETIONPENDING</code>.
 *
 * @return An int32_t containing the number of bytes read or an error code
 * from <code>pp_errors.h</code>.
 */
static int32_t ReadResponseBody(PP_Resource loader, void* buffer, int32_t bytes_to_read,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};


/**
 * FinishStreamingToFile() is used to wait for the response body to be
 * completely downloaded to the file provided by the GetBodyAsFileRef()
 * in the current <code>URLResponseInfo</code>. This function is only used if
 * <code>PP_URLREQUESTPROPERTY_STREAMTOFILE</code> was set on the
 * <code>URLRequestInfo</code> passed to Open().
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * asynchronous completion. This callback will run when body is downloaded
 * or an error occurs after FinishStreamingToFile() returns
 * <code>PP_OK_COMPLETIONPENDING</code>.
 *
 * @return An int32_t containing the number of bytes read or an error code
 * from <code>pp_errors.h</code>.
 */
static int32_t FinishStreamingToFile(PP_Resource loader, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Close is a pointer to a function used to cancel any pending IO and close
 * the <code>URLLoader</code> object. Any pending callbacks will still run,
 * reporting <code>PP_ERROR_ABORTED</code> if pending IO was interrupted.
 * It is NOT valid to call Open() again after a call to this function.
 *
 * <strong>Note:</strong> If the <code>URLLoader</code> object is destroyed
 * while it is still open, then it will be implicitly closed so you are not
 * required to call Close().
 *
 * @param[in] loader A <code>PP_Resource</code> corresponding to a
 * <code>URLLoader</code>.
 */
static void Close(PP_Resource loader)
{
    LOG_NP;
};


struct PPB_URLLoader_1_0 PPB_URLLoader_1_0_instance =
{
    .Create = Create,
    .IsURLLoader = IsURLLoader,
    .Open = Open,
    .FollowRedirect = FollowRedirect,
    .GetUploadProgress = GetUploadProgress,
    .GetDownloadProgress = GetDownloadProgress,
    .GetResponseInfo = GetResponseInfo,
    .ReadResponseBody = ReadResponseBody,
    .FinishStreamingToFile = FinishStreamingToFile,
    .Close = Close,
};
