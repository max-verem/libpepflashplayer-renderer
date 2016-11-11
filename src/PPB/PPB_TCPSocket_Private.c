#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_tcp_socket_private.h>

#include "log.h"

/**
 * Allocates a TCP socket resource.
 */
static PP_Resource Create(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

/**
 * Determines if a given resource is TCP socket.
 */
static PP_Bool IsTCPSocket(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

/**
 * Connects to a TCP port given as a host-port pair.
 * When a proxy server is used, |host| and |port| refer to the proxy server
 * instead of the destination server.
 */
static int32_t Connect(PP_Resource tcp_socket,
    const char* host, uint16_t port,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
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
static PP_Bool GetLocalAddress(PP_Resource tcp_socket, struct PP_NetAddress_Private* local_addr)
{
    LOG_NP;
    return 0;
};

/**
 * Gets the remote address of the socket, if it has been connected.
 * Returns PP_TRUE on success.
 */
static PP_Bool GetRemoteAddress(PP_Resource tcp_socket, struct PP_NetAddress_Private* remote_addr)
{
    LOG_NP;
    return 0;
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
static int32_t Read(PP_Resource tcp_socket,
    char* buffer, int32_t bytes_to_read,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Writes data to the socket. May perform a partial write. Returns the number
 * of bytes written or an error code.
 * This method won't write more than 1 megabyte, so if |bytes_to_write|
 * exceeds 1 megabyte, it will always perform a partial write.
 * Multiple outstanding write requests are not supported.
 */
static int32_t Write(PP_Resource tcp_socket,
    const char* buffer, int32_t bytes_to_write,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
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
    LOG_NP;
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
