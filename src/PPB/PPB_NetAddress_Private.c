#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_net_address_private.h>

#include "log.h"

static PP_Bool AreEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    LOG_NP;
    return 0;
};

static PP_Bool AreHostsEqual(const struct PP_NetAddress_Private* addr1, const struct PP_NetAddress_Private* addr2)
{
    LOG_NP;
    return 0;
};

static struct PP_Var Describe(PP_Module module, const struct PP_NetAddress_Private* addr, PP_Bool include_port)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static PP_Bool ReplacePort(const struct PP_NetAddress_Private* src_addr,
    uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    LOG_NP;
    return 0;
};

static void GetAnyAddress(PP_Bool is_ipv6, struct PP_NetAddress_Private* addr)
{
    LOG_NP;
};

static PP_NetAddressFamily_Private GetFamily(const struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

static uint16_t GetPort(const struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

static PP_Bool GetAddress(const struct PP_NetAddress_Private* addr, void* address, uint16_t address_size)
{
    LOG_NP;
    return 0;
};

static uint32_t GetScopeID(const struct PP_NetAddress_Private* addr)
{
    LOG_NP;
    return 0;
};

static void CreateFromIPv4Address(const uint8_t ip[4], uint16_t port, struct PP_NetAddress_Private* addr_out)
{
    LOG_NP;
};

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
