#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <curl/curl.h>
#include <errno.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_url_loader.h>

#include "log.h"
#include "res.h"

#include "PPB_Var.h"
#include "PPB_URLUtil_Dev.h"
#include "PPB_URLRequestInfo.h"
#include "PPB_URLLoader.h"
#include "PPB_MessageLoop.h"

static int URL_COMPONENT_EMPTY(const char* s, struct PP_URLComponent_Dev c)
{
    return (c.begin == 0 && c.len == -1)?1:0;
};

#define CHAR_EQ(IDX, U, L)      \
    u = s[c.begin + IDX];       \
    if(u != U && u != L)        \
        return 0;

static int URL_COMPONENT_IS_FILE(const char* s, struct PP_URLComponent_Dev c)
{
    char u;

    if(URL_COMPONENT_EMPTY(s, c) || c.len != 4)
        return 0;

    CHAR_EQ(0, 'F', 'f');
    CHAR_EQ(1, 'I', 'i');
    CHAR_EQ(2, 'L', 'l');
    CHAR_EQ(3, 'E', 'e');

    return 1;
};

static int URL_COMPONENT_IS_FTP(const char* s, struct PP_URLComponent_Dev c)
{
    char u;

    if(URL_COMPONENT_EMPTY(s, c) || c.len != 3)
        return 0;

    CHAR_EQ(0, 'F', 'f');
    CHAR_EQ(1, 'T', 't');
    CHAR_EQ(2, 'P', 'p');

    return 1;
};

static int URL_COMPONENT_IS_HTTP(const char* s, struct PP_URLComponent_Dev c)
{
    char u;

    if(URL_COMPONENT_EMPTY(s, c) || c.len != 4)
        return 0;

    CHAR_EQ(0, 'H', 'h');
    CHAR_EQ(1, 'T', 't');
    CHAR_EQ(2, 'T', 't');
    CHAR_EQ(3, 'P', 'p');

    return 1;
};

static int f_curl_global_init_done = 0;
#define CURL_GLOBAL_INIT                        \
    /* init curl lib */                         \
    if(!f_curl_global_init_done)                \
    {                                           \
        f_curl_global_init_done = 1;            \
        curl_global_init(CURL_GLOBAL_ALL);      \
    }

static size_t curl_read_from_buf_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    size_t n, l;
    url_loader_t* url_loader = (url_loader_t*)userdata;
    url_request_info_t* url_request_info = url_loader->url_request_info;

    /* left */
    n = size * nmemb;
    l = url_request_info->DataToBody.len - url_loader->body_pos;

    /* send data size */
    n = (n < l) ? n : l;

    /* copy */
    memcpy(ptr, url_request_info->DataToBody.data + url_loader->body_pos, n);
    LOG("{%d} url_loader->body_pos=%d, n=%d", url_loader->self, url_loader->body_pos, (int)n);
    url_loader->body_pos += n;

    return n;
};

static size_t curl_read_from_file_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    size_t n = 0;
    url_loader_t* url_loader = (url_loader_t*)userdata;
    url_request_info_t* url_request_info = url_loader->url_request_info;

    LOG_NP;

    return n;
};

static size_t curl_write_received_cb(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    size_t n = 0;
    url_loader_t* url_loader = (url_loader_t*)userdata;

    LOG("");

    if(!url_loader->writer)
    {
        char name[PATH_MAX];

        /* build temp rss filename */
        strncpy(name, "/tmp/PPB_URLLoader-XXXXXX", sizeof(name));
        mktemp(name);

        LOG("{%d} name=[%s]", url_loader->self, name);

        url_loader->writer = fopen(name, "wb");

        if(!url_loader->writer)
        {
            LOG("{%d} error=[%s]", url_loader->self, strerror(errno));
            n = CURL_READFUNC_ABORT;
            url_loader->state = URLLoader_ABORTED;
        }
        else
        {
            url_loader->reader = fopen(name, "rb");
            url_loader->state = URLLoader_DOWNLOADING;
            if(url_loader->header_data)
                url_loader->response.HEADERS = VarFromUtf8(url_loader->header_data, url_loader->header_len);

            PPB_MessageLoop_push(0, url_loader->callback, 0, PP_OK);
        };
    };

    if(url_loader->writer)
    {
        n = fwrite(ptr, size, nmemb, url_loader->writer) * size;
        fflush(url_loader->writer);

        if(ferror(url_loader->writer))
        {
            n = CURL_READFUNC_ABORT;
            LOG("{%d} ferror=[%s]", url_loader->self, strerror(errno));
        }
        else
            url_loader->bytes_received += n;
    };

    LOG("{%d} size=%lld, nmemb=%lld, n=%lld", url_loader->self, size, nmemb, n);

    return n;
};

static size_t curl_header_cb(char *buffer, size_t size, size_t nitems, void *userdata)
{
    size_t n = size * nitems;
    url_loader_t* url_loader = (url_loader_t*)userdata;

    if(!url_loader->header_data)
        url_loader->header_data = (char*)malloc(0);

    url_loader->header_data = (char*)realloc(url_loader->header_data, 1 + n + url_loader->header_len);
    memcpy(url_loader->header_data + url_loader->header_len, buffer, n);
    url_loader->header_len += n;
    url_loader->header_data[url_loader->header_len] = 0;

    LOG("{%d} %4d|%4d| %.*s", url_loader->self, (int)size, (int)nitems, n, buffer);

    return n;
};

static int curl_xferinfo_cb(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow)
{
    LOG("");

//    url_loader_t* url_loader = (url_loader_t*)clientp;
    return 0;
};

static int curl_progress_cb(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow)
{
    return curl_xferinfo_cb(clientp,
                  (curl_off_t)dltotal,
                  (curl_off_t)dlnow,
                  (curl_off_t)ultotal,
                  (curl_off_t)ulnow);
};

static void* curl_downloader(void* p)
{
    int r;
    CURL *curl;
    struct curl_slist *headers_list = NULL;
    char* curl_error_msg = NULL;
    url_loader_t* url_loader = (url_loader_t*)p;
    url_request_info_t* url_request_info = url_loader->url_request_info;

    LOG("{%d} url=[%s]", url_loader->self, url_loader->url);

    /* init curl lib */
    CURL_GLOBAL_INIT;

    /* init curl */
    curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url_loader->url);

    /* get verbose debug output please */
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    /* check if it post */
    if(url_request_info->props[PP_URLREQUESTPROPERTY_METHOD].type == PP_VARTYPE_STRING &&
        !strcasecmp("POST", VarToUtf8(url_request_info->props[PP_URLREQUESTPROPERTY_METHOD], NULL)))
    {
        LOG("{%d} POST", url_loader->self);
        curl_easy_setopt(curl, CURLOPT_POST, 1);
    };

    /* setup writer function */
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_received_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, url_loader);

    /* setup reader */
    if(url_request_info->DataToBody.data)
    {
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, curl_read_from_buf_cb);
        curl_easy_setopt(curl, CURLOPT_READDATA, url_loader);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, url_request_info->DataToBody.len);
    }
    else if(url_request_info->FileToBody.file_ref)
    {
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, curl_read_from_file_cb);
        curl_easy_setopt(curl, CURLOPT_READDATA, url_loader);
        headers_list = curl_slist_append(headers_list, "Transfer-Encoding: chunked");
    };

    /* progress */
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_error_msg = (char*)malloc(CURL_ERROR_SIZE));
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, curl_progress_cb);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, url_loader);
#if LIBCURL_VERSION_NUM >= 0x072000
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_xferinfo_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, url_loader);
#endif

    /* header received */
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, curl_header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, url_loader);

    /* setup headers */
    if(url_request_info->props[PP_URLREQUESTPROPERTY_HEADERS].type == PP_VARTYPE_STRING)
    {
        const char *h, *t;
        const char* headers = VarToUtf8(url_request_info->props[PP_URLREQUESTPROPERTY_HEADERS], NULL);

        for(h = headers; h;)
        {
            char
                *e,
                *sr = strchr(h, '\r'),
                *sn = strchr(h, '\n');

            if(sr && sn)
                t = (sr < sn)?sr:sn;
            else if(sr)
                t = sr;
            else if(sn)
                t = sn;
            else
                t = h + strlen(h);

            e = strndup(h, t - h);

            LOG("e=[%s]", e);

            if(*e)
            {
                headers_list = curl_slist_append(headers_list, e);
                h = t + 1;
                LOG("curl_slist_append(%s)", e);
            }
            else
                h = NULL;

            if(e)
                free(e);
        };
    };

    if(headers_list)
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers_list);

    /* run transfer */
    r = curl_easy_perform(curl);
    LOG("curl_easy_perform return %d", r);
    if(r)
    {
        PPB_MessageLoop_push(0, url_loader->callback, 0, PP_ERROR_FAILED);

        LOG("{%d} curl_error_msg=[%s]", url_loader->self, curl_error_msg);

        url_loader->state = URLLoader_ABORTED;
    }
    else
        url_loader->state = URLLoader_DONE;

    if(url_loader->writer)
    {
        fclose(url_loader->writer);
        url_loader->writer = NULL;
    };

    curl_easy_cleanup(curl);

    curl_slist_free_all(headers_list);

    return NULL;
};


static void Destructor(url_loader_t* ctx)
{
    LOG("{%d}", ctx->self);
    LOG_NP;
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
    url_loader->self = res;

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
    int curl = 0;
    struct PP_Var url_var;
    struct PP_URLComponents_Dev comp;
    url_loader_t* url_loader = (url_loader_t*)res_private(loader);
    url_request_info_t* url_request_info = (url_request_info_t*)res_private(request_info);

    LOG("{%d}", loader);

    /* check if it not busy */
    if(url_loader->state != URLLoader_NONE)
    {
        LOG("{%d} BUSY, url_loader->state=%d", loader, url_loader->state);
        return PP_ERROR_INPROGRESS;
    };

    /* parse url */
    url_var = url_request_info->props[PP_URLREQUESTPROPERTY_URL];
    if(url_var.type != PP_VARTYPE_STRING)
    {
        LOG("{%d} url is not a string", loader);
        return PP_ERROR_BADARGUMENT;
    };

    /* check if supported */
    url_loader->url = VarToUtf8(url_var, NULL);
    uriparser_parse(url_loader->url, &comp);
    if
    (
        (
            URL_COMPONENT_EMPTY(url_loader->url, comp.scheme)
            ||
            URL_COMPONENT_IS_FILE(url_loader->url, comp.scheme)
        )
        &&
        (!URL_COMPONENT_EMPTY(url_loader->url, comp.path))
    )
    {
        LOG("{%d} will try localfile", loader);
        curl = 0;
    }
    else if
    (
        URL_COMPONENT_IS_FTP(url_loader->url, comp.scheme)
        ||
        URL_COMPONENT_IS_HTTP(url_loader->url, comp.scheme)
    )
    {
        LOG("{%d} will try curl", loader);
        curl = 1;
    }
    else
    {
        LOG("{%d} NOTHING", loader);
        return PP_ERROR_BADARGUMENT;
    };

    /* save request */
    url_loader->request_info = request_info;
    res_add_ref(request_info);
    url_loader->url_request_info = url_request_info;

    url_loader->total_bytes_to_be_received = 0;

    /* */
    if(curl)
    {
        url_loader->state = URLLoader_CONNECTING;
        url_loader->callback = callback;

        pthread_create(&url_loader->th, NULL, curl_downloader, url_loader);

        return PP_OK_COMPLETIONPENDING;
    }
    else
    {
        char* filename;

        filename = calloc(1, comp.path.len + 1);
        memcpy(filename, url_loader->url + comp.path.begin, comp.path.len);
        url_loader->reader = fopen(filename, "rb");
        free(filename);

        url_loader->response.HEADERS = VarFromUtf8("", 0);

        if(url_loader->reader)
        {
            struct stat st;

            url_loader->response.STATUSCODE = PP_MakeInt32(200);

            memset(&st, 0, sizeof(st));
            stat(filename, &st);

            url_loader->bytes_received = st.st_size;
        }
        else
            url_loader->response.STATUSCODE = PP_MakeInt32(404);


        url_loader->state = URLLoader_DONE;

        return PP_OK;;
    };
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
    url_loader_t* url_loader = (url_loader_t*)res_private(loader);

    LOG("{%d}", loader);

    if(!url_loader->reader)
        return 0;

    *bytes_received = url_loader->bytes_received;
    *total_bytes_to_be_received = url_loader->total_bytes_to_be_received;

    return 1;
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
    LOG("{%d}", loader);
    res_add_ref(loader);
    return loader;
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
    int r;
    url_loader_t* url_loader;

    LOG("{%d}", loader);

    url_loader = (url_loader_t*)res_private(loader);

    LOG("{%d}, url_loader=%p", loader, url_loader);

    if(url_loader->reader)
    {
        r = fread(buffer, 1, bytes_to_read, url_loader->reader);
        if(-1 == r)
            r = PP_ERROR_FAILED;
        else if (0 == r)
        {
            LOG_NP;
//            r = PP_OK_COMPLETIONPENDING;
        }
    }
    else
        r = PP_ERROR_FAILED;

    LOG("{%d}, r=%d, bytes_to_read=%d", loader, r, bytes_to_read);

    return r;
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
