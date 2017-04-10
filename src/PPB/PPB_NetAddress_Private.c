#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_net_address_private.h>

#include <errno.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include "PPB_Var.h"

#include "log.h"

struct PPB_NetAddress_Private_1_1 PPB_NetAddress_Private_1_1_instance;

/**
 * Returns PP_TRUE if the two addresses are equal (host and port).
 */
static PP_Bool AreEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    struct sockaddr_in *a1 = (struct sockaddr_in *)addr1->data;
    struct sockaddr_in *a2 = (struct sockaddr_in *)addr2->data;

    return (a1->sin_port == a2->sin_port) && PPB_NetAddress_Private_1_1_instance.AreHostsEqual(addr1, addr2);
};

/**
 * Returns PP_TRUE if the two addresses refer to the same host.
 */
static PP_Bool AreHostsEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    struct sockaddr_in *a1 = (struct sockaddr_in *)addr1->data;
    struct sockaddr_in *a2 = (struct sockaddr_in *)addr2->data;
    return a1->sin_addr.s_addr == a2->sin_addr.s_addr;
};

/**
 * Returns a human-readable description of the network address, optionally
 * including the port (e.g., "192.168.0.1", "192.168.0.1:99", or "[::1]:80"),
 * or an undefined var on failure.
 */
static struct PP_Var Describe(PP_Module module, const struct PP_NetAddress_Private* addr, PP_Bool include_port)
{
    char buf[32];
    struct sockaddr_in *s = (struct sockaddr_in *)addr->data;

    if(include_port)
        snprintf(buf, sizeof(buf), "%s:%d", inet_ntoa(s->sin_addr), ntohs(s->sin_port));
    else
        snprintf(buf, sizeof(buf), "%s", inet_ntoa(s->sin_addr));

    return VarFromUtf8_c(buf);
};

/**
 * Replaces the port in the given source address. Returns PP_TRUE on success.
 */
static PP_Bool ReplacePort(const struct PP_NetAddress_Private* src_addr,
    uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    struct sockaddr_in * s = (struct sockaddr_in *)addr_out->data;

    memcpy(addr_out, src_addr, sizeof(struct PP_NetAddress_Private));

    s->sin_port = htons(port);

    return PP_TRUE;
};

/**
 * Gets the "any" address (for IPv4 or IPv6); for use with UDP Bind.
 */
static void GetAnyAddress(PP_Bool is_ipv6, struct PP_NetAddress_Private* addr)
{
    LOG_NP;
};

/**
 * Gets the address family.
 */
static PP_NetAddressFamily_Private GetFamily(const struct PP_NetAddress_Private* addr)
{
    return PP_NETADDRESSFAMILY_PRIVATE_IPV4;
};

/**
 * Gets the port. The port is returned in host byte order.
 */
static uint16_t GetPort(const struct PP_NetAddress_Private* addr)
{
    struct sockaddr_in *s = (struct sockaddr_in *)addr->data;
    return ntohs(s->sin_port);
};

/**
 * Gets the address. The output, address, must be large enough for the
 * current socket family. The output will be the binary representation of an
 * address for the current socket family. For IPv4 and IPv6 the address is in
 * network byte order. PP_TRUE is returned if the address was successfully
 * retrieved.
 */
static PP_Bool GetAddress(const struct PP_NetAddress_Private* addr, void* address, uint16_t address_size)
{
    struct sockaddr_in *s = (struct sockaddr_in *)addr->data;

    LOG_N("address_size=%d, sizeof(s->sin_addr.s_addr)=%d", (int)address_size, (int)sizeof(s->sin_addr.s_addr));

    memcpy(address, &s->sin_addr.s_addr, (address_size > sizeof(s->sin_addr.s_addr))?sizeof(s->sin_addr.s_addr):address_size);

    return PP_TRUE;
};

/**
 * Returns ScopeID for IPv6 addresses or 0 for IPv4.
 */
static uint32_t GetScopeID(const struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

/**
 * Creates NetAddress with the specified IPv4 address and port
 * number.
 */
static void CreateFromIPv4Address(const uint8_t ip[4], uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    LOG_NP;
};

/**
 * Creates NetAddress with the specified IPv6 address, scope_id and
 * port number.
 */
static void CreateFromIPv6Address(const uint8_t ip[16],uint32_t scope_id, uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    LOG_NP;
};


struct PPB_NetAddress_Private_1_1 PPB_NetAddress_Private_1_1_instance =
{
    .AreEqual = AreEqual,
    .AreHostsEqual = AreHostsEqual,
    .Describe = Describe,
    .ReplacePort = ReplacePort,
    .GetAnyAddress = GetAnyAddress,
    .GetFamily = GetFamily,
    .GetPort = GetPort,
    .GetAddress = GetAddress,
    .GetScopeID = GetScopeID,
    .CreateFromIPv4Address = CreateFromIPv4Address,
    .CreateFromIPv6Address = CreateFromIPv6Address,
};
