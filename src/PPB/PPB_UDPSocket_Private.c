#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_udp_socket_private.h>

#include "log.h"

/**
 * Creates a UDP socket resource.
 */
static PP_Resource Create(PP_Instance instance_id)
{
    LOG_NP;
    return 0;
};

/**
 * Determines if a given resource is a UDP socket.
 */
static PP_Bool IsUDPSocket(PP_Resource resource_id)
{
    LOG_NP;
    return 0;
};

/**
 * Sets a socket feature to |udp_socket|. Should be called before
 * Bind(). Possible values for |name|, |value| and |value|'s type
 * are described in PP_UDPSocketFeature_Private description. If no
 * error occurs, returns PP_OK. Otherwise, returns
 * PP_ERROR_BADRESOURCE (if bad |udp_socket| provided),
 * PP_ERROR_BADARGUMENT (if bad name/value/value's type provided)
 * or PP_ERROR_FAILED in the case of internal errors.
 */
static int32_t SetSocketFeature(PP_Resource udp_socket,
    PP_UDPSocketFeature_Private name, struct PP_Var value)
{
    LOG_NP;
    return 0;
};


/* Creates a socket and binds to the address given by |addr|. */
static int32_t Bind(PP_Resource udp_socket,
    const struct PP_NetAddress_Private* addr,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/* Returns the address that the socket has bound to.  A successful
 * call to Bind must be called first. Returns PP_FALSE if Bind
 * fails, or if Close has been called.
 */
static PP_Bool GetBoundAddress(PP_Resource udp_socket, struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

/* Performs a non-blocking recvfrom call on socket.
 * Bind must be called first. |callback| is invoked when recvfrom
 * reads data.  You must call GetRecvFromAddress to recover the
 * address the data was retrieved from.
 */
static int32_t RecvFrom(PP_Resource udp_socket,
    char* buffer, int32_t num_bytes,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/* Upon successful completion of RecvFrom, the address that the data
 * was received from is stored in |addr|.
 */
static PP_Bool GetRecvFromAddress(PP_Resource udp_socket, struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

/* Performs a non-blocking sendto call on the socket created and
 * bound(has already called Bind).  The callback |callback| is
 * invoked when sendto completes.
 */
static int32_t SendTo(PP_Resource udp_socket,
    const char* buffer, int32_t num_bytes,
    const struct PP_NetAddress_Private* addr,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/* Cancels all pending reads and writes, and closes the socket. */
static void Close(PP_Resource udp_socket)
{
    LOG_NP;
};

struct PPB_UDPSocket_Private_0_4 PPB_UDPSocket_Private_0_4_instance =
{
    .Create = Create,
    .IsUDPSocket = IsUDPSocket,
    .SetSocketFeature = SetSocketFeature,
    .Bind = Bind,
    .GetBoundAddress = GetBoundAddress,
    .RecvFrom = RecvFrom,
    .GetRecvFromAddress = GetRecvFromAddress,
    .SendTo = SendTo,
    .Close = Close,
};
