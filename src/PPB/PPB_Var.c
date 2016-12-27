#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_var.h>
#include <ppapi/c/dev/ppb_var_deprecated.h>

#include "log.h"
#include "res.h"

#define REFERENCABLE(V) \
( \
    V.type == PP_VARTYPE_STRING         || \
    V.type == PP_VARTYPE_OBJECT         || \
    V.type == PP_VARTYPE_ARRAY          || \
    V.type == PP_VARTYPE_DICTIONARY     || \
    V.type == PP_VARTYPE_ARRAY_BUFFER   || \
    V.type == PP_VARTYPE_RESOURCE          \
)

/**
 * AddRef() adds a reference to the given var. If this is not a refcounted
 * object, this function will do nothing so you can always call it no matter
 * what the type.
 *
 * @param[in] var A <code>PP_Var</code> that will have a reference added.
 */
void PPB_Var_AddRef(struct PP_Var var)
{
    int r;

    if(!REFERENCABLE(var))
        return;

    r = res_add_ref(var.value.as_id);

    LOG("{%d} r=%d", (int)var.value.as_id, r);
};

/**
 * Release() removes a reference to given var, deleting it if the internal
 * reference count becomes 0. If the <code>PP_Var</code> is of type
 * <code>PP_VARTYPE_RESOURCE</code>,
 * it will implicitly release a reference count on the
 * <code>PP_Resource</code> (equivalent to PPB_Core::ReleaseResource()).
 *
 * If the given var is not a refcounted object, this function will do nothing
 * so you can always call it no matter what the type.
 *
 * @param[in] var A <code>PP_Var</code> that will have a reference removed.
 */
void PPB_Var_Release(struct PP_Var var)
{
    int r;

    if(!REFERENCABLE(var))
        return;

    r = res_release(var.value.as_id);

    LOG("{%d} r=%d", (int)var.value.as_id, r);

    if(!r)
    {
        LOG_TD;
    };
};

/**
 * VarFromUtf8() creates a string var from a string. The string must be
 * encoded in valid UTF-8 and is NOT NULL-terminated, the length must be
 * specified in <code>len</code>. It is an error if the string is not
 * valid UTF-8.
 *
 * If the length is 0, the <code>*data</code> pointer will not be dereferenced
 * and may be <code>NULL</code>. Note, however if length is 0, the
 * "NULL-ness" will not be preserved, as VarToUtf8() will never return
 * <code>NULL</code> on success, even for empty strings.
 *
 * The resulting object will be a refcounted string object. It will be
 * AddRef'ed for the caller. When the caller is done with it, it should be
 * Released.
 *
 * On error (basically out of memory to allocate the string, or input that
 * is not valid UTF-8), this function will return a Null var.
 *
 * @param[in] data A string
 * @param[in] len The length of the string.
 *
 * @return A <code>PP_Var</code> structure containing a reference counted
 * string object.
 */
struct PP_Var VarFromUtf8(const char* data, uint32_t len)
{
    void* dst;
    struct PP_Var var;

    var.type = PP_VARTYPE_STRING;
    var.value.as_id = res_create(len + 1, NULL, NULL);

    dst = (char*)res_private(var.value.as_id);
    if(len && data)
        memcpy(dst, data, len);

    LOG1("{%d} data=[%s], len=%d", (int)var.value.as_id, data, len);

    return var;
};


/**
 * VarToUtf8() converts a string-type var to a char* encoded in UTF-8. This
 * string is NOT NULL-terminated. The length will be placed in
 * <code>*len</code>. If the string is valid but empty the return value will
 * be non-NULL, but <code>*len</code> will still be 0.
 *
 * If the var is not a string, this function will return NULL and
 * <code>*len</code> will be 0.
 *
 * The returned buffer will be valid as long as the underlying var is alive.
 * If the instance frees its reference, the string will be freed and the
 * pointer will be to arbitrary memory.
 *
 * @param[in] var A PP_Var struct containing a string-type var.
 * @param[in,out] len A pointer to the length of the string-type var.
 *
 * @return A char* encoded in UTF-8.
 */
const char* VarToUtf8(struct PP_Var var, uint32_t* len)
{
    const char* r;

    LOG("var.value.as_id=%d", (int)var.value.as_id);

    if(var.type != PP_VARTYPE_STRING)
    {
        LOG("var.type != PP_VARTYPE_STRING");
        if(len)
            *len = 0;
        return NULL;
    };

    r = (char*)res_private(var.value.as_id);
    if(len)
        *len = res_private_size(var.value.as_id) - 1;

    LOG1("r=[%s], *len=%d, strlen=%zd", r, (len)?(*len):0, strlen(r));

    return r;
};


/**
 * Converts a resource-type var to a <code>PP_Resource</code>.
 *
 * @param[in] var A <code>PP_Var</code> struct containing a resource-type var.
 *
 * @return A <code>PP_Resource</code> retrieved from the var, or 0 if the var
 * is not a resource. The reference count of the resource is incremented on
 * behalf of the caller.
 */
static PP_Resource VarToResource(struct PP_Var var)
{
    PP_Resource res;

    if(var.type != PP_VARTYPE_RESOURCE)
        return 0;

    res = var.value.as_id;

    res_add_ref(res);

    LOG("res=%d", res);

    return res;
};


/**
 * Creates a new <code>PP_Var</code> from a given resource. Implicitly adds a
 * reference count on the <code>PP_Resource</code> (equivalent to
 * PPB_Core::AddRefResource(resource)).
 *
 * @param[in] resource A <code>PP_Resource</code> to be wrapped in a var.
 *
 * @return A <code>PP_Var</code> created for this resource, with type
 * <code>PP_VARTYPE_RESOURCE</code>. The reference count of the var is set to
 * 1 on behalf of the caller.
 */
struct PP_Var VarFromResource(PP_Resource resource)
{
    struct PP_Var var;

    var.type = PP_VARTYPE_RESOURCE;
    var.value.as_id = resource;
    res_add_ref(resource);

    LOG("resource=%d", resource);

    return var;
};

struct PPB_Var_1_2 PPB_Var_1_2_instance =
{
    .AddRef = PPB_Var_AddRef,
    .Release = PPB_Var_Release,
    .VarFromUtf8 = VarFromUtf8,
    .VarToUtf8 = VarToUtf8,
    .VarToResource = VarToResource,
    .VarFromResource = VarFromResource,
};


