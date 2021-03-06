#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_tcp_socket_private.h>

#include <pthread.h>
#include <errno.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include "PPB_MessageLoop.h"

#include "res.h"
#include "log.h"

static int32_t errno2pp(int e)
{
    int r = PP_ERROR_FAILED;

    switch(e)
    {
        case EACCES:
        case EPERM: return PP_ERROR_NOACCESS;
        case EADDRINUSE: return PP_ERROR_ADDRESS_IN_USE;
        case ECONNREFUSED: return PP_ERROR_CONNECTION_REFUSED;
        case ENETUNREACH: return PP_ERROR_ADDRESS_UNREACHABLE;
        case ETIMEDOUT: return PP_ERROR_CONNECTION_TIMEDOUT;
        case ENOTCONN: return PP_ERROR_CONNECTION_CLOSED;
        case ECONNRESET: return PP_ERROR_CONNECTION_RESET;
        case EAGAIN:
        case EBADF: return PP_ERROR_FAILED;
        default: LOG_E("unhandled %d (%s)", e, strerror(e));
    };

    return r;
};

struct PPB_TCPSocket_Private_0_5 PPB_TCPSocket_Private_0_5_instance;

typedef struct ppb_tcpsocket_private_desc
{
    uint64_t async, asyncs;
    PP_Resource self;
    PP_Instance instance_id;
    pthread_mutex_t lock;
    int s;
} ppb_tcpsocket_private_t;

static void ppb_tcpsocket_private_destructor(ppb_tcpsocket_private_t* ctx)
{
    LOG_D("{%d}, ctx=%p, async=%d, asyncs=%d", ctx->self, ctx, (int)ctx->async, (int)ctx->asyncs);

    pthread_mutex_lock(&ctx->lock);

    while(ctx->asyncs)
    {
        LOG_D("{%d}, asyncs=%d", ctx->self, (int)ctx->asyncs);
        pthread_mutex_unlock(&ctx->lock);
        usleep(1000);
        pthread_mutex_lock(&ctx->lock);
    };

    pthread_mutex_unlock(&ctx->lock);

    LOG_D("{%d}, ctx=%p, async=%d, asyncs=%d", ctx->self, ctx, (int)ctx->async, (int)ctx->asyncs);

    /* destroy mutex */
    pthread_mutex_destroy(&ctx->lock);
};


/**
 * Allocates a TCP socket resource.
 */
static PP_Resource Create(PP_Instance instance)
{
    ppb_tcpsocket_private_t* ctx;
    int res = res_create(sizeof(ppb_tcpsocket_private_t), &PPB_TCPSocket_Private_0_5_instance, (res_destructor_t)ppb_tcpsocket_private_destructor);

    /* get private data */
    ctx = (ppb_tcpsocket_private_t*)res_private(res);

    LOG_D("res=%d, ctx=%p", res, ctx);

    /* setup internals */
    ctx->instance_id = instance;
    ctx->self = res;

    /* create mutex mutex */
    pthread_mutex_init(&ctx->lock, NULL);

    return res;
};

/**
 * Determines if a given resource is TCP socket.
 */
static PP_Bool IsTCPSocket(PP_Resource resource)
{
    return (res_interface(resource) == &PPB_TCPSocket_Private_0_5_instance);
};

/**
 * Connects to a TCP port given as a host-port pair.
 * When a proxy server is used, |host| and |port| refer to the proxy server
 * instead of the destination server.
 */
typedef struct ArgsDesc
{
    struct PP_CompletionCallback callback;
    uint64_t async;
    ppb_tcpsocket_private_t* ctx;
    char* host;
    uint16_t port;
    char* buffer;
    int32_t bytes;
    pthread_t th;
} Args_t;

static void* ConnectProcAsync(void* a)
{
    int r;
    struct hostent *host_ip = NULL;
    Args_t* args = (Args_t*)a;


    /* resolv hostname */
    host_ip = gethostbyname(args->host);
    if(!host_ip)
    {
        LOG_E("res=%d, gethostbyname(%s) failed", args->ctx->self, args->host);

        /* send callback */
        pthread_mutex_lock(&args->ctx->lock);
        args->ctx->asyncs--;
        PPB_MessageLoop_push(0, args->callback, 0, PP_ERROR_NAME_NOT_RESOLVED);
        pthread_mutex_unlock(&args->ctx->lock);
    }
    else
    {
        int s;

        pthread_mutex_lock(&args->ctx->lock);
//        args->ctx->hoste = hostbuf;
        pthread_mutex_unlock(&args->ctx->lock);

        /* create communication socket */
        s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if(s < 0)
        {
            s = errno;

            LOG_E("res=%d, socket() failed, errno=%d", args->ctx->self, s);

            /* send callback */
            pthread_mutex_lock(&args->ctx->lock);
            args->ctx->asyncs--;
            PPB_MessageLoop_push(0, args->callback, 0, errno2pp(s));
            pthread_mutex_unlock(&args->ctx->lock);
        }
        else
        {
            struct sockaddr_in s_remote;

            /* prepare address */
            s_remote.sin_family = AF_INET;
            s_remote.sin_addr.s_addr = inet_addr(inet_ntoa(*(struct in_addr*)(host_ip->h_addr_list[0])));
            s_remote.sin_port = htons((unsigned short)args->port);

            /* connect */
            r = connect(s, (struct sockaddr*)&s_remote, sizeof(s_remote));
            if(r < 0)
            {
                r = errno;

                LOG_E("res=%d, connect(%s:%d) failed, errno=%d (%s)", args->ctx->self, args->host, args->port, r, strerror(r));

                /* send callback */
                pthread_mutex_lock(&args->ctx->lock);
                args->ctx->asyncs--;
                PPB_MessageLoop_push(0, args->callback, 0, errno2pp(r));
                pthread_mutex_unlock(&args->ctx->lock);

                close(s);
            }
            else
            {
                int f;

                /* set socket option */
                f = 1; r = setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (char*)&f, sizeof(f));

                /* setup SO_KEEPALIVE for socket */
                f = 1; r = setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (char*)&f, sizeof(f));

                LOG_N("res=%d, socket(%s:%d)=%d", args->ctx->self, args->host, args->port, s);

                /* send callback */
                pthread_mutex_lock(&args->ctx->lock);
                args->ctx->s = s;
                args->ctx->asyncs--;
                PPB_MessageLoop_push(0, args->callback, 0, PP_OK);
                pthread_mutex_unlock(&args->ctx->lock);
            };
        };
    };

    free(args->host);
    free(args);

    return NULL;
};

static int32_t Connect(PP_Resource tcp_socket,
    const char* host, uint16_t port,
    struct PP_CompletionCallback callback)
{
    Args_t* args;
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    LOG_N("res=%d, socket=%d", tcp_socket, ctx->s);

//    PPB_TCPSocket_Private_0_5_instance.Disconnect(tcp_socket);

    pthread_mutex_lock(&ctx->lock);
    args = (Args_t*)calloc(1, sizeof(Args_t));
    args->host = strdup(host);
    args->port = port;
    args->callback = callback;
    args->ctx = ctx;
    args->async = ++args->ctx->async;
    ctx->asyncs++;
    pthread_create(&args->th, NULL, ConnectProcAsync, args);
    pthread_detach(args->th);
    pthread_mutex_unlock(&ctx->lock);

    return PP_OK_COMPLETIONPENDING;
};

/**
 * Same as Connect(), but connecting to the address given by |addr|. A typical
 * use-case would be for reconnections.
 */
static int32_t ConnectWithNetAddress(PP_Resource tcp_socket,
    const struct PP_NetAddress_Private* addr, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Gets the local address of the socket, if it has been connected.
 * Returns PP_TRUE on success.
 */
static PP_Bool GetLocalAddress(PP_Resource tcp_socket, struct PP_NetAddress_Private* addr)
{
    int r;
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    pthread_mutex_lock(&ctx->lock);

    if(ctx->s)
    {
        struct sockaddr_in s;
        socklen_t len = sizeof(s);

        r = getsockname(ctx->s, (struct sockaddr *)&s, &len);
        if(r < 0)
        {
            r = errno;
            LOG_E("res=%d, getsockname(%d)=%d (%s)", tcp_socket, ctx->s, r, strerror(r));
            r = PP_FALSE;
        }
        else
        {
            LOG_N("local %s:%d", inet_ntoa(s.sin_addr), ntohs(s.sin_port));

            /* copy */
            addr->size = sizeof(s);
            memcpy(addr->data, &s, addr->size);
            r = PP_TRUE;
        }
    }
    else
        r = PP_FALSE;

    pthread_mutex_unlock(&ctx->lock);

    return r;
};

/**
 * Gets the remote address of the socket, if it has been connected.
 * Returns PP_TRUE on success.
 */
static PP_Bool GetRemoteAddress(PP_Resource tcp_socket, struct PP_NetAddress_Private* addr)
{
    int r;
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    pthread_mutex_lock(&ctx->lock);

    if(ctx->s)
    {
        struct sockaddr_in s;
        socklen_t len = sizeof(s);

        r = getpeername(ctx->s, (struct sockaddr *)&s, &len);
        if(r < 0)
        {
            r = errno;
            LOG_E("res=%d, getsockname(%d)=%d (%s)", tcp_socket, ctx->s, r, strerror(r));
            r = PP_FALSE;
        }
        else
        {
            LOG_N("remote %s:%d", inet_ntoa(s.sin_addr), ntohs(s.sin_port));

            /* copy */
            addr->size = sizeof(s);
            memcpy(addr->data, &s, addr->size);
            r = PP_TRUE;
        }
    }
    else
        r = PP_FALSE;

    pthread_mutex_unlock(&ctx->lock);

    return r;
};

/**
 * Does SSL handshake and moves to sending and receiving encrypted data. The
 * socket must have been successfully connected. |server_name| will be
 * compared with the name(s) in the server's certificate during the SSL
 * handshake. |server_port| is only used to identify an SSL server in the SSL
 * session cache.
 * When a proxy server is used, |server_name| and |server_port| refer to the
 * destination server.
 * If the socket is not connected, or there are pending read/write requests,
 * SSLHandshake() will fail without starting a handshake. Otherwise, any
 * failure during the handshake process will cause the socket to be
 * disconnected.
 */
static int32_t SSLHandshake(PP_Resource tcp_socket,
    const char* server_name, uint16_t server_port,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Returns the server's <code>PPB_X509Certificate_Private</code> for a socket
 * connection if an SSL connection has been established using
 * <code>SSLHandshake</code>. If no SSL connection has been established, a
 * null resource is returned.
 */
static PP_Resource GetServerCertificate(PP_Resource tcp_socket)
{
    LOG_NP;
    return 0;
};

/**
 * NOTE: This function is not implemented and will return
 * <code>PP_FALSE</code>.
 * Adds a trusted/untrusted chain building certificate to be used for this
 * connection. The <code>certificate</code> must be a
 * <code>PPB_X509Certificate_Private<code>. <code>PP_TRUE</code> is returned
 * upon success.
 */
static PP_Bool AddChainBuildingCertificate(PP_Resource tcp_socket,
    PP_Resource certificate, PP_Bool is_trusted)
{
    LOG_NP;
    return 0;
};

/**
 * Reads data from the socket. The size of |buffer| must be at least as large
 * as |bytes_to_read|. May perform a partial read. Returns the number of bytes
 * read or an error code. If the return value is 0, then it indicates that
 * end-of-file was reached.
 * This method won't return more than 1 megabyte, so if |bytes_to_read|
 * exceeds 1 megabyte, it will always perform a partial read.
 * Multiple outstanding read requests are not supported.
 */
static void* ReadProcAsync(void* a)
{
    int s;
    Args_t* args = (Args_t*)a;

    LOG_N("ent");

    pthread_mutex_lock(&args->ctx->lock);
    s = args->ctx->s;
    pthread_mutex_unlock(&args->ctx->lock);

    if(s <= 0)
    {
        /* send callback */
        pthread_mutex_lock(&args->ctx->lock);
        args->ctx->asyncs--;
        PPB_MessageLoop_push(0, args->callback, 0, PP_ERROR_CONNECTION_CLOSED);
        pthread_mutex_unlock(&args->ctx->lock);
    }
    else
    {
        int r;

        r = read(s, args->buffer, args->bytes);

        if(r < 0)
        {
            r = errno;

            LOG_E("res=%d, read(%d) failed, errno=%d", args->ctx->self, s, r);

            /* send callback */
            pthread_mutex_lock(&args->ctx->lock);
            args->ctx->asyncs--;
            PPB_MessageLoop_push(0, args->callback, 0, errno2pp(r));
            pthread_mutex_unlock(&args->ctx->lock);
        }
        else
        {
            /* send callback */
            pthread_mutex_lock(&args->ctx->lock);
            args->ctx->asyncs--;
            PPB_MessageLoop_push(0, args->callback, 0, r);
            pthread_mutex_unlock(&args->ctx->lock);
        };
    };

    free(args);

    LOG_N("ex");

    return NULL;
};

static int32_t Read(PP_Resource tcp_socket,
    char* buffer, int32_t bytes_to_read,
    struct PP_CompletionCallback callback)
{
    Args_t* args;
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    LOG_N("ent");

    pthread_mutex_lock(&ctx->lock);
    args = (Args_t*)calloc(1, sizeof(Args_t));
    args->buffer = buffer;
    args->bytes = bytes_to_read;
    args->callback = callback;
    args->ctx = ctx;
    args->async = ++args->ctx->async;
    ctx->asyncs++;
    pthread_create(&args->th, NULL, ReadProcAsync, args);
    pthread_detach(args->th);
    pthread_mutex_unlock(&ctx->lock);

    LOG_N("ex");

    return PP_OK_COMPLETIONPENDING;
};

/**
 * Writes data to the socket. May perform a partial write. Returns the number
 * of bytes written or an error code.
 * This method won't write more than 1 megabyte, so if |bytes_to_write|
 * exceeds 1 megabyte, it will always perform a partial write.
 * Multiple outstanding write requests are not supported.
 */
static void* WriteProcAsync(void* a)
{
    int s;
    Args_t* args = (Args_t*)a;

    pthread_mutex_lock(&args->ctx->lock);
    s = args->ctx->s;
    pthread_mutex_unlock(&args->ctx->lock);

    if(s <= 0)
    {
        /* send callback */
        pthread_mutex_lock(&args->ctx->lock);
        args->ctx->asyncs--;
        PPB_MessageLoop_push(0, args->callback, 0, PP_ERROR_CONNECTION_CLOSED);
        pthread_mutex_unlock(&args->ctx->lock);
    }
    else
    {
        int r;

        r = write(s, args->buffer, args->bytes);

        if(r < 0)
        {
            r = errno;

            LOG_E("res=%d, write(%d) failed, errno=%d", args->ctx->self, s, r);

            /* send callback */
            pthread_mutex_lock(&args->ctx->lock);
            args->ctx->asyncs--;
            PPB_MessageLoop_push(0, args->callback, 0, errno2pp(r));
            pthread_mutex_unlock(&args->ctx->lock);
        }
        else
        {
            /* send callback */
            pthread_mutex_lock(&args->ctx->lock);
            args->ctx->asyncs--;
            PPB_MessageLoop_push(0, args->callback, 0, r);
            pthread_mutex_unlock(&args->ctx->lock);
        };
    };

    free(args);

    return NULL;
};

static int32_t Write(PP_Resource tcp_socket,
    const char* buffer, int32_t bytes_to_write,
    struct PP_CompletionCallback callback)
{
    Args_t* args;
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    pthread_mutex_lock(&ctx->lock);
    args = (Args_t*)calloc(1, sizeof(Args_t));
    args->buffer = (char*)buffer;
    args->bytes = bytes_to_write;
    args->callback = callback;
    args->ctx = ctx;
    args->async = ++args->ctx->async;
    ctx->asyncs++;
    pthread_create(&args->th, NULL, WriteProcAsync, args);
    pthread_detach(args->th);
    pthread_mutex_unlock(&ctx->lock);

    return PP_OK_COMPLETIONPENDING;
};

/**
 * Cancels any IO that may be pending, and disconnects the socket. Any pending
 * callbacks will still run, reporting PP_Error_Aborted if pending IO was
 * interrupted. It is NOT valid to call Connect() again after a call to this
 * method. Note: If the socket is destroyed when it is still connected, then
 * it will be implicitly disconnected, so you are not required to call this
 * method.
 */
static void Disconnect(PP_Resource tcp_socket)
{
    ppb_tcpsocket_private_t* ctx = (ppb_tcpsocket_private_t*)res_private(tcp_socket);

    if(!ctx)
    {
        LOG_E("res=%d, ctx=NULL", tcp_socket);
        return;
    };

    pthread_mutex_lock(&ctx->lock);
    LOG_N("res=%d, socket=%d", tcp_socket, ctx->s);
    if(ctx->s)
    {
        shutdown(ctx->s, SHUT_RDWR);
        close(ctx->s);
    };
    ctx->s = 0;
    pthread_mutex_unlock(&ctx->lock);
};

/**
 * Sets an option on |tcp_socket|.  Supported |name| and |value| parameters
 * are as described for PP_TCPSocketOption_Private.  |callback| will be
 * invoked with PP_OK if setting the option succeeds, or an error code
 * otherwise. The socket must be connection before SetOption is called.
 */
static int32_t SetOption(PP_Resource tcp_socket,
    PP_TCPSocketOption_Private name,  struct PP_Var value,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};


struct PPB_TCPSocket_Private_0_5 PPB_TCPSocket_Private_0_5_instance =
{
    .Create = Create,
    .IsTCPSocket = IsTCPSocket,
    .Connect = Connect,
    .ConnectWithNetAddress = ConnectWithNetAddress,
    .GetLocalAddress = GetLocalAddress,
    .GetRemoteAddress = GetRemoteAddress,
    .SSLHandshake = SSLHandshake,
    .GetServerCertificate = GetServerCertificate,
    .AddChainBuildingCertificate = AddChainBuildingCertificate,
    .Read = Read,
    .Write = Write,
    .Disconnect = Disconnect,
    .SetOption = SetOption,
};
