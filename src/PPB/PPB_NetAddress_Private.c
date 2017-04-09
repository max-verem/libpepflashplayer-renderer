#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_net_address_private.h>

#include "log.h"


#if 0
/**
 * This is an opaque type holding a network address. Plugins must
 * never access members of this struct directly.
 */
struct PP_NetAddress_Private {
  uint32_t size;
  char data[128];
};
#endif

/**
 * Returns PP_TRUE if the two addresses are equal (host and port).
 */
static PP_Bool AreEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    LOG_NP;
    return 0;
};

/**
 * Returns PP_TRUE if the two addresses refer to the same host.
 */
static PP_Bool AreHostsEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    LOG_NP;
    return 0;
};

/**
 * Returns a human-readable description of the network address, optionally
 * including the port (e.g., "192.168.0.1", "192.168.0.1:99", or "[::1]:80"),
 * or an undefined var on failure.
 */
static struct PP_Var Describe(PP_Module module, const struct PP_NetAddress_Private* addr, PP_Bool include_port)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

/**
 * Replaces the port in the given source address. Returns PP_TRUE on success.
 */
static PP_Bool ReplacePort(const struct PP_NetAddress_Private* src_addr,
    uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    LOG_NP;
    return 0;
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
    LOG_NP;
    return 0;
};

/**
 * Gets the port. The port is returned in host byte order.
 */
static uint16_t GetPort(const struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
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
    LOG_NP;
    return 0;
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
