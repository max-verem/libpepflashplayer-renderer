#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_net_address.h>

#if 0
#include <errno.h>
#include <netdb.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#endif

#include "PPB_Var.h"

#include "res.h"
#include "log.h"

struct PPB_NetAddress_1_0 PPB_NetAddress_1_0_instance;

typedef struct ppb_net_address_desc
{
    PP_Resource self;
    struct PP_NetAddress_IPv6 v6;
    struct PP_NetAddress_IPv4 v4;
    PP_NetAddress_Family f;
} ppb_net_address_t;

static void ppb_net_address_destructor(ppb_net_address_t* ctx)
{
    LOG_D("{%d}", ctx->self);
};

  /**
   * Creates a <code>PPB_NetAddress</code> resource with the specified IPv4
   * address.
   *
   * @param[in] instance A <code>PP_Instance</code> identifying one instance of
   * a module.
   * @param[in] ipv4_addr An IPv4 address.
   *
   * @return A <code>PP_Resource</code> representing the same address as
   * <code>ipv4_addr</code> or 0 on failure.
   */
static PP_Resource CreateFromIPv4Address(PP_Instance instance, const struct PP_NetAddress_IPv4* ipv4_addr)
{
    PP_Resource res;
    ppb_net_address_t* ctx;

    res = res_create(sizeof(ppb_net_address_t), &PPB_NetAddress_1_0_instance, (res_destructor_t)ppb_net_address_destructor);

    LOG_N("res={%d}", (int)res);

    ctx = (ppb_net_address_t*)res_private(res);
    ctx->self = res;
    ctx->v4 = *ipv4_addr;
    ctx->f = PP_NETADDRESS_FAMILY_IPV4;

    return res;
};

  /**
   * Creates a <code>PPB_NetAddress</code> resource with the specified IPv6
   * address.
   *
   * @param[in] instance A <code>PP_Instance</code> identifying one instance of
   * a module.
   * @param[in] ipv6_addr An IPv6 address.
   *
   * @return A <code>PP_Resource</code> representing the same address as
   * <code>ipv6_addr</code> or 0 on failure.
   */
static PP_Resource CreateFromIPv6Address(PP_Instance instance, const struct PP_NetAddress_IPv6* ipv6_addr)
{
    PP_Resource res;
    ppb_net_address_t* ctx;

    res = res_create(sizeof(ppb_net_address_t), &PPB_NetAddress_1_0_instance, (res_destructor_t)ppb_net_address_destructor);

    LOG_N("res={%d}", (int)res);

    ctx = (ppb_net_address_t*)res_private(res);
    ctx->self = res;
    ctx->v6 = *ipv6_addr;
    ctx->f = PP_NETADDRESS_FAMILY_IPV6;

    return res;
};

  /**
   * Determines if a given resource is a network address.
   *
   * @param[in] resource A <code>PP_Resource</code> to check.
   *
   * @return <code>PP_TRUE</code> if the input is a <code>PPB_NetAddress</code>
   * resource; <code>PP_FALSE</code> otherwise.
   */
static PP_Bool IsNetAddress(PP_Resource resource)
{
    return (res_interface(resource) == &PPB_NetAddress_1_0_instance);
};

  /**
   * Gets the address family.
   *
   * @param[in] addr A <code>PP_Resource</code> corresponding to a network
   * address.
   *
   * @return The address family on success;
   * <code>PP_NETADDRESS_FAMILY_UNSPECIFIED</code> on failure.
   */
static PP_NetAddress_Family GetFamily(PP_Resource addr)
{
    if(PPB_NetAddress_1_0_instance.IsNetAddress(addr))
    {
        ppb_net_address_t* ctx = (ppb_net_address_t*)res_private(addr);
        return ctx->f;
    }
    else
        return PP_NETADDRESS_FAMILY_UNSPECIFIED;
};


  /**
   * Returns a human-readable description of the network address. The
   * description is in the form of host [ ":" port ] and conforms to
   * http://tools.ietf.org/html/rfc3986#section-3.2 for IPv4 and IPv6 addresses
   * (e.g., "192.168.0.1", "192.168.0.1:99", or "[::1]:80").
   *
   * @param[in] addr A <code>PP_Resource</code> corresponding to a network
   * address.
   * @param[in] include_port Whether to include the port number in the
   * description.
   *
   * @return A string <code>PP_Var</code> on success; an undefined
   * <code>PP_Var</code> on failure.
   */
static struct PP_Var DescribeAsString(PP_Resource addr, PP_Bool include_port)
{
    char buf[64];
    ppb_net_address_t* ctx = (ppb_net_address_t*)res_private(addr);

    if(ctx->f == PP_NETADDRESS_FAMILY_IPV4)
    {
        if(include_port)
            snprintf(buf, sizeof(buf), "%d.%d.%d.%d:%d", ctx->v4.addr[0], ctx->v4.addr[1], ctx->v4.addr[2], ctx->v4.addr[3], ctx->v4.port);
        else
            snprintf(buf, sizeof(buf), "%d.%d.%d.%d", ctx->v4.addr[0], ctx->v4.addr[1], ctx->v4.addr[2], ctx->v4.addr[3]);
    }
    else if(ctx->f == PP_NETADDRESS_FAMILY_IPV6)
    {
        snprintf(buf, sizeof(buf), "<IPV6>");
        LOG_NP;
    }
    else
        snprintf(buf, sizeof(buf), "<UNSPECIFIED>");

    return VarFromUtf8_c(buf);
};

  /**
   * Fills a <code>PP_NetAddress_IPv4</code> structure if the network address is
   * of <code>PP_NETADDRESS_FAMILY_IPV4</code> address family.
   * Note that passing a network address of
   * <code>PP_NETADDRESS_FAMILY_IPV6</code> address family will fail even if the
   * address is an IPv4-mapped IPv6 address.
   *
   * @param[in] addr A <code>PP_Resource</code> corresponding to a network
   * address.
   * @param[out] ipv4_addr A <code>PP_NetAddress_IPv4</code> structure to store
   * the result.
   *
   * @return A <code>PP_Bool</code> value indicating whether the operation
   * succeeded.
   */
static PP_Bool DescribeAsIPv4Address(PP_Resource addr,
                                   struct PP_NetAddress_IPv4* ipv4_addr)
{
    ppb_net_address_t* ctx = (ppb_net_address_t*)res_private(addr);

    if(ctx->f != PP_NETADDRESS_FAMILY_IPV4)
        return PP_FALSE;

    *ipv4_addr = ctx->v4;

    return PP_TRUE;
};


  /**
   * Fills a <code>PP_NetAddress_IPv6</code> structure if the network address is
   * of <code>PP_NETADDRESS_FAMILY_IPV6</code> address family.
   * Note that passing a network address of
   * <code>PP_NETADDRESS_FAMILY_IPV4</code> address family will fail - this
   * method doesn't map it to an IPv6 address.
   *
   * @param[in] addr A <code>PP_Resource</code> corresponding to a network
   * address.
   * @param[out] ipv6_addr A <code>PP_NetAddress_IPv6</code> structure to store
   * the result.
   *
   * @return A <code>PP_Bool</code> value indicating whether the operation
   * succeeded.
   */
static PP_Bool DescribeAsIPv6Address(PP_Resource addr,
                                   struct PP_NetAddress_IPv6* ipv6_addr)
{
    ppb_net_address_t* ctx = (ppb_net_address_t*)res_private(addr);

    if(ctx->f != PP_NETADDRESS_FAMILY_IPV6)
        return PP_FALSE;

    *ipv6_addr = ctx->v6;

    return PP_TRUE;
};

struct PPB_NetAddress_1_0 PPB_NetAddress_1_0_instance =
{
    .CreateFromIPv4Address = CreateFromIPv4Address,
    .CreateFromIPv6Address = CreateFromIPv6Address,
    .IsNetAddress = IsNetAddress,
    .GetFamily = GetFamily,
    .DescribeAsString = DescribeAsString,
    .DescribeAsIPv4Address = DescribeAsIPv4Address,
    .DescribeAsIPv6Address = DescribeAsIPv6Address,
};
