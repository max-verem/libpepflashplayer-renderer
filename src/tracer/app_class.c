#include "log.h"
#include "app.h"
#include "app_class.h"

#include "PPB_Var.h"
#include "PPB.h"

/**
 * |name| is guaranteed to be an integer or string type var. Exception is
 * guaranteed non-NULL. An integer is used for |name| when implementing
 * array access into the object. This test should only return true for
 * properties that are not methods.  Use HasMethod() to handle methods.
 */
static bool HasProperty(void* object,
                      struct PP_Var name,
                      struct PP_Var* exception)
{
    LOG_NP;
    return 0;
};

/**
 * |name| is guaranteed to be a string-type. Exception is guaranteed non-NULL.
 * If the method does not exist, return false and don't set the exception.
 * Errors in this function will probably not occur in general usage, but
 * if you need to throw an exception, still return false.
 */
static bool HasMethod(void* object,
                    struct PP_Var name,
                    struct PP_Var* exception)
{
    LOG_NP;
    return 0;
};

/**
 * |name| is guaranteed to be a string-type or an integer-type var. Exception
 * is guaranteed non-NULL. An integer is used for |name| when implementing
 * array access into the object. If the property does not exist, set the
 * exception and return a var of type Void. A property does not exist if
 * a call HasProperty() for the same |name| would return false.
 */
static struct PP_Var GetProperty(void* object,
                               struct PP_Var name,
                               struct PP_Var* exception)
{
    uint32_t len;
    app_t* app = (app_t*)object;

    if(name.type != PP_VARTYPE_STRING)
    {
        LOG_N("object=%p", object);
        PPB_Var_Dump("name", name);
        return PP_MakeNull(); //PP_MakeUndefined();
    };

    const char* name_c = VarToUtf8(name, &len);

    LOG_N("name=[%s]", name_c);

    if(!strcmp(name_c, "top") || !strcmp(name_c, "location"))
        return app->inst->window_instance_object;

    if(!strcmp(name_c, "href"))
        return VarFromUtf8_c("/");

    LOG_N("not handled=[%s]", name_c);

    return PP_MakeUndefined();;
};


/**
 * Exception is guaranteed non-NULL.
 *
 * This should include all enumerable properties, including methods. Be sure
 * to set |*property_count| to 0 and |properties| to NULL in all failure
 * cases, these should never be unset when calling this function. The
 * pointers passed in are guaranteed not to be NULL, so you don't have to
 * NULL check them.
 *
 * If you have any properties, allocate the property array with
 * PPB_Core.MemAlloc(sizeof(PP_Var) * property_count) and add a reference
 * to each property on behalf of the caller. The caller is responsible for
 * Release()ing each var and calling PPB_Core.MemFree on the property pointer.
 */
static void GetAllPropertyNames(void* object,
                              uint32_t* property_count,
                              struct PP_Var** properties,
                              struct PP_Var* exception)
{
    LOG_NP;
};

/**
  * |name| is guaranteed to be an integer or string type var. Exception is
  * guaranteed non-NULL.
  */
static void SetProperty(void* object,
                      struct PP_Var name,
                      struct PP_Var value,
                      struct PP_Var* exception)
{
    LOG_NP;
};

/**
 * |name| is guaranteed to be an integer or string type var. Exception is
 * guaranteed non-NULL.
 */
static void RemoveProperty(void* object,
                         struct PP_Var name,
                         struct PP_Var* exception)
{
    LOG_NP;
};

  // TODO(brettw) need native array access here.

/**
 * |name| is guaranteed to be a string type var. Exception is guaranteed
 * non-NULL
 */
static struct PP_Var Call(void* object,
                        struct PP_Var method_name,
                        uint32_t argc,
                        struct PP_Var* argv,
                        struct PP_Var* exception)
{
    int i;
    LOG_N("argc=%d", argc);
    PPB_Var_Dump("Call(method_name", method_name);
    for(i = 0; i < argc; i++)
    {
        LOG_N("i=%d", i);
        PPB_Var_Dump("argv", argv[i]);
    };
    return PP_MakeUndefined();
};


/** Exception is guaranteed non-NULL. */
static struct PP_Var Construct(void* object,
                             uint32_t argc,
                             struct PP_Var* argv,
                             struct PP_Var* exception)
{
    LOG_NP;
    return PP_MakeUndefined();
};

/**
 * Called when the reference count of the object reaches 0. Normally, plugins
 * would free their internal data pointed to by the |object| pointer.
 */
static void Deallocate(void* object)
{
    LOG_NP;
};

const struct PPP_Class_Deprecated app_class_struct =
{
    .HasProperty = HasProperty,
    .HasMethod = HasMethod,
    .GetProperty = GetProperty,
    .GetAllPropertyNames = GetAllPropertyNames,
    .SetProperty = SetProperty,
    .RemoveProperty = RemoveProperty,
    .Call = Call,
    .Construct = Construct,
    .Deallocate = Deallocate,
};
