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

#include "impl/PPB_Var.h"

typedef struct ppb_var_deprecated_desc
{
    const struct PPP_Class_Deprecated* object_class;
    void* object_data;
} ppb_var_deprecated_t;

static void ppb_var_deprecated_destructor(ppb_var_deprecated_t* p)
{
};

/**
 * Returns true if the property with the given name exists on the given
 * object, false if it does not. Methods are also counted as properties.
 *
 * The name can either be a string or an integer var. It is an error to pass
 * another type of var as the name.
 *
 * If you pass an invalid name or object, the exception will be set (if it is
 * non-NULL, and the return value will be false).
 */
static bool HasProperty(struct PP_Var object, struct PP_Var name,
    struct PP_Var* exception)
{
    LOG_NP;
    return 0;
};


/**
 * Identical to HasProperty, except that HasMethod additionally checks if the
 * property is a function.
 */
static bool HasMethod(struct PP_Var object, struct PP_Var name,
    struct PP_Var* exception)
{
    LOG_NP;
    return 0;
};

/**
 * Returns the value of the given property. If the property doesn't exist, the
 * exception (if non-NULL) will be set and a "Void" var will be returned.
 */
static struct PP_Var GetProperty(struct PP_Var object,  struct PP_Var name,
    struct PP_Var* exception)
{
    char* q;
    uint32_t len;

    LOG_NP;
    q = VarToUtf8(name, &len);
    LOG("name=[%s]", q);

    return PP_MakeInt32(0);
};

/**
 * Retrieves all property names on the given object. Property names include
 * methods.
 *
 * If there is a failure, the given exception will be set (if it is non-NULL).
 * On failure, |*properties| will be set to NULL and |*property_count| will be
 * set to 0.
 *
 * A pointer to the array of property names will be placesd in |*properties|.
 * The caller is responsible for calling Release() on each of these properties
 * (as per normal refcounted memory management) as well as freeing the array
 * pointer with PPB_Core.MemFree().
 *
 * This function returns all "enumerable" properties. Some JavaScript
 * properties are "hidden" and these properties won't be retrieved by this
 * function, yet you can still set and get them.
 *
 * Example:
 * <pre>uint32_t count;
 * PP_Var* properties;
 * ppb_var.GetAllPropertyNames(object, &count, &properties);
 *
 * ...use the properties here...
 *
 * for (uint32_t i = 0; i < count; i++)
 * ppb_var.Release(properties[i]);
 * ppb_core.MemFree(properties); </pre>
 */
static void GetAllPropertyNames(struct PP_Var object,
    uint32_t* property_count, struct PP_Var** properties,
    struct PP_Var* exception)
{
    LOG_NP;
};

/**
 * Sets the property with the given name on the given object. The exception
 * will be set, if it is non-NULL, on failure.
 */
static void SetProperty(struct PP_Var object, struct PP_Var name, struct PP_Var value,
    struct PP_Var* exception)
{
    LOG_NP;
};

/**
 * Removes the given property from the given object. The property name must
 * be an string or integer var, using other types will throw an exception
 * (assuming the exception pointer is non-NULL).
 */
static void RemoveProperty(struct PP_Var object, struct PP_Var name,
    struct PP_Var* exception)
{
    LOG_NP;
};


// TODO(brettw) need native array access here.

/**
 * Invoke the function |method_name| on the given object. If |method_name|
 * is a Null var, the default method will be invoked, which is how you can
 * invoke function objects.
 *
 * Unless it is type Null, |method_name| must be a string. Unlike other
 * Var functions, integer lookup is not supported since you can't call
 * functions on integers in JavaScript.
 *
 * Pass the arguments to the function in order in the |argv| array, and the
 * number of arguments in the |argc| parameter. |argv| can be NULL if |argc|
 * is zero.
 *
 * Example:
 * Call(obj, VarFromUtf8("DoIt"), 0, NULL, NULL) = obj.DoIt() in JavaScript.
 * Call(obj, PP_MakeNull(), 0, NULL, NULL) = obj() in JavaScript.
 */
static struct PP_Var Call(struct PP_Var object,
    struct PP_Var method_name, uint32_t argc, struct PP_Var* argv,
    struct PP_Var* exception)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


/**
 * Invoke the object as a constructor.
 *
 * For example, if |object| is |String|, this is like saying |new String| in
 * JavaScript.
 */
static struct PP_Var Construct(struct PP_Var object,
    uint32_t argc, struct PP_Var* argv,
    struct PP_Var* exception)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


/**
 * If the object is an instance of the given class, then this method returns
 * true and sets *object_data to the value passed to CreateObject provided
 * object_data is non-NULL. Otherwise, this method returns false.
 */
static bool IsInstanceOf(struct PP_Var var,
    const struct PPP_Class_Deprecated* object_class, void** object_data)
{
    LOG_NP;
    return 0;
};


/**
 * Creates an object that the plugin implements. The plugin supplies a
 * pointer to the class interface it implements for that object, and its
 * associated internal data that represents that object. This object data
 * must be unique among all "live" objects.
 *
 * The returned object will have a reference count of 1. When the reference
 * count reached 0, the class' Destruct function will be called.
 *
 * On failure, this will return a null var. This probably means the module
 * was invalid.
 *
 * Example: Say we're implementing a "Point" object.
 * <pre>void PointDestruct(void* object) {
 * delete (Point*)object;
 * }
 *
 * const PPP_Class_Deprecated point_class = {
 * ... all the other class functions go here ...
 * &PointDestruct
 * };
 *
 ** The plugin's internal object associated with the point.
 * class Point {
 * ...
 * };
 *
 * PP_Var MakePoint(int x, int y) {
 * return CreateObject(&point_class, new Point(x, y));
 * }</pre>
 */
static struct PP_Var CreateObject(PP_Instance instance,
    const struct PPP_Class_Deprecated* object_class, void* object_data)
{
    struct PP_Var var;
    ppb_var_deprecated_t* dst;

    LOG("object_class=%p, object_data=%p", object_class, object_data);

    var.type = PP_VARTYPE_OBJECT;
    var.value.as_id = res_create(sizeof(ppb_var_deprecated_t), NULL, (res_destructor_t)ppb_var_deprecated_destructor);

    dst = (ppb_var_deprecated_t*)res_private(var.value.as_id);
    dst->object_class = object_class;
    dst->object_data = object_data;

    return var;
};


// Like CreateObject but takes a module. This will be deleted when all callers
// can be changed to use the PP_Instance CreateObject one.
static struct PP_Var CreateObjectWithModuleDeprecated(PP_Module module,
    const struct PPP_Class_Deprecated* object_class, void* object_data)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static struct PP_Var VarFromUtf8_(PP_Module module, const char* data, uint32_t len)
{
    LOG("");
    return VarFromUtf8(data, len);
};

struct PPB_Var_Deprecated PPB_Var_Deprecated_instance =
{
    .AddRef = PPB_Var_AddRef,
    .Release = PPB_Var_Release,
    .VarFromUtf8 = VarFromUtf8_,
    .VarToUtf8 = VarToUtf8,
    .HasProperty = HasProperty,
    .HasMethod = HasMethod,
    .GetProperty = GetProperty,
    .GetAllPropertyNames = GetAllPropertyNames,
    .SetProperty = SetProperty,
    .RemoveProperty = RemoveProperty,
    .Call = Call,
    .Construct = Construct,
    .IsInstanceOf = IsInstanceOf,
    .CreateObject = CreateObject,
    .CreateObjectWithModuleDeprecated = CreateObjectWithModuleDeprecated,
};
