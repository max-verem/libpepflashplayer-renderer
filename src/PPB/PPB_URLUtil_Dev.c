#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_url_util_dev.h>

#include "log.h"
#include "res.h"
#include "instance.h"
#include "PPB_Var.h"

#include <uriparser/Uri.h>

void uriparser_parse(const char* url, struct PP_URLComponents_Dev* comp)
{
    int r;
    UriUriA uri;
    UriParserStateA state;
    struct PP_URLComponent_Dev und = {0, -1};

    LOG1("url=[%s]", url);

    memset(&uri, 0, sizeof(uri));
    memset(&state, 0, sizeof(state));

    state.uri = &uri;
    r = uriParseUriA(&state, url);
    LOG1("uriParseUriA(%p=%s)=%d", url, url, r);

    if(uri.scheme.first && uri.scheme.afterLast)
    {
        comp->scheme.begin = uri.scheme.first - url;
        comp->scheme.len = uri.scheme.afterLast - uri.scheme.first;
    }
    else
        comp->scheme = und;
    LOG1("sheme=[%d,%d]", comp->scheme.begin, comp->scheme.len);

    LOG1("uri.hostText.afterLast=%p, uri.hostText.first=%p", uri.hostText.afterLast, uri.hostText.first);
    if(uri.hostText.first && (uri.hostText.afterLast != uri.hostText.first))
    {
        comp->host.begin = uri.hostText.first - url;
        comp->host.len = uri.hostText.afterLast - uri.hostText.first;
    }
    else
        comp->host = und;
    LOG1("host=[%d,%d]", comp->host.begin, comp->host.len);

    if(uri.portText.first)
    {
        comp->port.begin = uri.portText.first - url;
        comp->port.len = uri.portText.afterLast - uri.portText.first;
    }
    else
        comp->port = und;
    LOG1("port=[%d,%d]", comp->port.begin, comp->port.len);

    if(uri.query.first)
    {
        comp->query.begin = uri.query.first - url;
        comp->query.len = uri.query.afterLast - uri.query.first;
    }
    else
        comp->query = und;

    LOG1("query=[%d,%d]", comp->query.begin, comp->query.len);

    if(uri.fragment.first)
    {
        comp->ref.begin = uri.fragment.first - url;
        comp->ref.len = uri.fragment.afterLast - uri.fragment.first;
    }
    else
        comp->ref = und;

    LOG1("ref=[%d,%d]", comp->ref.begin, comp->ref.len);


    if(uri.userInfo.first)
    {
        char* tmp = strchr(uri.userInfo.first, ':');

        LOG1("uri.userInfo.first=%p, uri.userInfo.afterLast=%p, tmp=%p", uri.userInfo.first, uri.userInfo.afterLast, tmp);

        comp->username.begin = uri.userInfo.first - url;

        if(tmp && tmp < uri.userInfo.afterLast)
        {
            comp->username.len = tmp - uri.userInfo.first;

            comp->password.begin = tmp + 1 - url;
            comp->password.len = uri.userInfo.afterLast - (tmp + 1);
        }
        else
        {
            comp->username.len = uri.userInfo.afterLast - uri.userInfo.first;
            comp->password = und;
        };
    }
    else
    {
        comp->password = und;
        comp->username = und;
    };

    LOG1("username=[%d,%d]", comp->username.begin, comp->username.len);
    LOG1("password=[%d,%d]", comp->password.begin, comp->password.len);

    LOG1("uri.pathHead->text.first=%p, uri.pathTail->text.afterLast=%p",
        uri.pathHead->text.first, uri.pathTail->text.afterLast);
    if(uri.pathHead && (uri.pathHead->text.first != uri.pathHead->text.afterLast))
    {
        comp->path.begin = uri.pathHead->text.first - url - 1 ;

        if(uri.pathTail && (uri.pathTail->text.first != uri.pathTail->text.afterLast))
            comp->path.len = uri.pathTail->text.afterLast - uri.pathHead->text.first + 1;
        else
            comp->path.len = strlen(url) - comp->path.begin + 1;
    }
    else
        comp->path = und;

    LOG1("path=[%d,%d]", comp->path.begin, comp->path.len);


    uriFreeUriMembersA(&uri);
};

void uriparser_parse_var(struct PP_Var url, struct PP_URLComponents_Dev* comp)
{
    uriparser_parse(VarToUtf8(url, NULL), comp);
};

/*
 * Canonicalizes the given URL string according to the rules of the host
 * browser. If the URL is invalid or the var is not a string, this will
 * return a Null var and the components structure will be unchanged.
 *
 * The components pointer, if non-NULL and the canonicalized URL is valid,
 * will identify the components of the resulting URL. Components may be NULL
 * to specify that no component information is necessary.
 */
static struct PP_Var Canonicalize(struct PP_Var url, struct PP_URLComponents_Dev* components)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

/*
 *Resolves the given URL relative to the given base URL. The resulting URL
 *is returned as a string. If the resolution is invalid or either of the
 *inputs are not strings, a Null var will be returned. The resulting URL
 *will also be canonicalized according to the rules of the browser.
 *
 *Note that the "relative" URL may in fact be absolute, in which case it
 *will be returned. This function is identical to resolving the full URL
 *for an <a href="..."> on a web page. Attempting to resolve a relative URL
 *on a base URL that doesn't support this (e.g. "data") will fail and will
 *return a Null var, unless the relative URL is itself absolute.
 *
 *The components pointer, if non-NULL and the canonicalized URL is valid,
 *will identify the components of the resulting URL. Components may be NULL
 *to specify that no component information is necessary.
 */
static struct PP_Var ResolveRelativeToURL(struct PP_Var base_url,
    struct PP_Var relative_string, struct PP_URLComponents_Dev* components)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


/*
 *Identical to ResolveRelativeToURL except that the base URL is the base
 *URL of the document containing the given plugin instance.
 *
 *Danger: This will be identical to resolving a relative URL on the page,
 *and might be overridden by the page to something different than its actual
 *URL via the <base> tag. Therefore, resolving a relative URL of "" won't
 *necessarily give you the URL of the page!
 */
static struct PP_Var ResolveRelativeToDocument(PP_Instance instance,
    struct PP_Var relative_string, struct PP_URLComponents_Dev* components)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

/*
 * Checks whether the given two URLs are in the same security origin. Returns
 * FALSE if either of the URLs are invalid.
 */
static PP_Bool IsSameSecurityOrigin(struct PP_Var url_a, struct PP_Var url_b)
{
    LOG_NP;
    return 0;
};

/*
 * Checks whether the document hosting the given plugin instance can access
 * the given URL according to the same origin policy of the browser. Returns
 * PP_FALSE if the instance or the URL is invalid.
 */
static PP_Bool DocumentCanRequest(PP_Instance instance, struct PP_Var url)
{
    LOG_NP;
    return 0;
};

/*
 * Checks whether the document containing the |active| plugin instance can
 * access the document containing the |target| plugin instance according to
 * the security policy of the browser. This includes the same origin policy
 * and any cross-origin capabilities enabled by the document. If either of
 * the plugin instances are invalid, returns PP_FALSE.
 */
static PP_Bool DocumentCanAccessDocument(PP_Instance active, PP_Instance target)
{
    LOG_NP;
    return 0;
};


/*
 * Returns the URL for the document. This is a safe way to retrieve
 * window.location.href.
 * The components pointer, if non-NULL and the canonicalized URL is valid,
 * will identify the components of the resulting URL. Components may be NULL
 * to specify that no component information is necessary.
 */
static struct PP_Var GetDocumentURL(PP_Instance instance, struct PP_URLComponents_Dev* components)
{
    instance_t* inst = (instance_t*)res_private(instance);

    LOG_D("[%s]", inst->paths.DocumentURL);

    if(components)
        uriparser_parse(inst->paths.DocumentURL, components);

    return VarFromUtf8_c(inst->paths.DocumentURL);
};

/*
 * Returns the Source URL for the plugin. This returns the URL that would be
 * streamed to the plugin if it were a NPAPI plugin. This is usually the src
 * attribute on the <embed> element, but the rules are obscure and different
 * based on whether the plugin is loaded from an <embed> element or an
 * <object> element.
 * The components pointer, if non-NULL and the canonicalized URL is valid,
 * will identify the components of the resulting URL. Components may be NULL
 * to specify that no component information is necessary.
 */
static struct PP_Var GetPluginInstanceURL(PP_Instance instance, struct PP_URLComponents_Dev* components)
{
    instance_t* inst = (instance_t*)res_private(instance);

    LOG_D("[%s]", inst->paths.PluginInstanceURL);

    if(components)
        uriparser_parse(inst->paths.PluginInstanceURL, components);

    return VarFromUtf8_c(inst->paths.PluginInstanceURL);
};


/*
 * Returns the Referrer URL of the HTTP request that loaded the plugin. This
 * is the value of the 'Referer' header of the request. An undefined value
 * means the 'Referer' header was absent.
 * The components pointer, if non-NULL and the canonicalized URL is valid,
 * will identify the components of the resulting URL. Components may be NULL
 * to specify that no component information is necessary.
 */
struct PP_Var GetPluginReferrerURL(PP_Instance instance, struct PP_URLComponents_Dev* components)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


struct PPB_URLUtil_Dev_0_7 PPB_URLUtil_Dev_0_7_instance =
{
    .Canonicalize = Canonicalize,
    .ResolveRelativeToURL = ResolveRelativeToURL,
    .ResolveRelativeToDocument = ResolveRelativeToDocument,
    .IsSameSecurityOrigin = IsSameSecurityOrigin,
    .DocumentCanRequest = DocumentCanRequest,
    .DocumentCanAccessDocument = DocumentCanAccessDocument,
    .GetDocumentURL = GetDocumentURL,
    .GetPluginInstanceURL = GetPluginInstanceURL,
    .GetPluginReferrerURL = GetPluginReferrerURL,
};
