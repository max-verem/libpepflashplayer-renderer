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

    LOG_T("{%d} r=%d", (int)var.value.as_id, r);
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

    LOG_T("{%d} r=%d", (int)var.value.as_id, r);

    if(!r)
    {
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

    LOG_T("{%d} data=[%s], len=%d", (int)var.value.as_id, data, len);

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

    LOG_T("var.value.as_id=%d", (int)var.value.as_id);

    if(var.type != PP_VARTYPE_STRING)
    {
        LOG_E("var.type != PP_VARTYPE_STRING");
        if(len)
            *len = 0;
        return NULL;
    };

    r = (char*)res_private(var.value.as_id);
    if(len)
        *len = res_private_size(var.value.as_id) - 1;

    LOG_T("r=[%s], *len=%d, strlen=%zd", r, (len)?(*len):0, strlen(r));

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

    LOG_T("res=%d", res);

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

    LOG_T("resource=%d", resource);

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

static const char* var_types[] =
{
    "PP_VARTYPE_UNDEFINED",
    "PP_VARTYPE_NULL",
    "PP_VARTYPE_BOOL",
    "PP_VARTYPE_INT32",
    "PP_VARTYPE_DOUBLE",
    "PP_VARTYPE_STRING",
    "PP_VARTYPE_OBJECT",
    "PP_VARTYPE_ARRAY",
    "PP_VARTYPE_DICTIONARY",
    "PP_VARTYPE_ARRAY_BUFFER",
    "PP_VARTYPE_RESOURCE",
};

void PPB_Var_Dump(const char* name, struct PP_Var var)
{
    uint32_t len;
    LOG_N("%s.type=%s", name, var_types[var.type]);

    switch(var.type)
    {
        /**
         * A boolean value, use the <code>as_bool</code> member of the var.
         */
        case PP_VARTYPE_BOOL:
            LOG_N("%s.value.as_bool=%d", name, (int)var.value.as_bool);
            break;

        /**
         * A 32-bit integer value. Use the <code>as_int</code> member of the var.
         */
        case PP_VARTYPE_INT32:
            LOG_N("%s.value.as_int=%d", name, (int)var.value.as_int);
            break;

        /**
         * A double-precision floating point value. Use the <code>as_double</code>
         * member of the var.
         */
        case PP_VARTYPE_DOUBLE:
            LOG_N("%s.value.as_double=%lf", name, var.value.as_double);
            break;

        /**
         * The Var represents a string. The <code>as_id</code> field is used to
         * identify the string, which may be created and retrieved from the
         * <code>PPB_Var</code> interface. These objects are reference counted, so
         * AddRef() and Release() must be used properly to avoid memory leaks.
         */
        case PP_VARTYPE_STRING:
            LOG_N("%s=%s", name, VarToUtf8(var, &len));
            break;

         /**
          * Represents a JavaScript object. This vartype is not currently usable
          * from modules, although it is used internally for some tasks. These objects
          * are reference counted, so AddRef() and Release() must be used properly to
          * avoid memory leaks.
          */
        case PP_VARTYPE_OBJECT:

        /**
         * Represents an array of Vars. The <code>as_id</code> field is used to
         * identify the array, which may be created and manipulated from the
         * <code>PPB_VarArray</code> interface. These objects are reference counted,
         * so AddRef() and Release() must be used properly to avoid memory leaks.
         */
        case PP_VARTYPE_ARRAY:

        /**
         * Represents a mapping from strings to Vars. The <code>as_id</code> field is
         * used to identify the dictionary, which may be created and manipulated from
         * the <code>PPB_VarDictionary</code> interface. These objects are reference
         * counted, so AddRef() and Release() must be used properly to avoid memory
         * leaks.
         */
        case PP_VARTYPE_DICTIONARY:

        /**
         * ArrayBuffer represents a JavaScript ArrayBuffer. This is the type which
         * represents Typed Arrays in JavaScript. Unlike JavaScript 'Array', it is
         * only meant to contain basic numeric types, and is always stored
         * contiguously. See PPB_VarArrayBuffer_Dev for functions special to
         * ArrayBuffer vars. These objects are reference counted, so AddRef() and
         * Release() must be used properly to avoid memory leaks.
         */
        case PP_VARTYPE_ARRAY_BUFFER:

        /**
         * This type allows the <code>PP_Var</code> to wrap a <code>PP_Resource
         * </code>. This can be useful for sending or receiving some types of
         * <code>PP_Resource</code> using <code>PPB_Messaging</code> or
         * <code>PPP_Messaging</code>.
         *
         * These objects are reference counted, so AddRef() and Release() must be used
         * properly to avoid memory leaks. Under normal circumstances, the
         * <code>PP_Var</code> will implicitly hold a reference count on the
         * <code>PP_Resource</code> on your behalf. For example, if you call
         * VarFromResource(), it implicitly calls PPB_Core::AddRefResource() on the
         * <code>PP_Resource</code>. Likewise, PPB_Var::Release() on a Resource
         * <code>PP_Var</code> will invoke PPB_Core::ReleaseResource() when the Var
         * reference count goes to zero.
         */
        case PP_VARTYPE_RESOURCE:
            break;

        default:
            LOG_N("%s.type=%d (UNHANDLED)", name, var.type);
            break;
    };
};