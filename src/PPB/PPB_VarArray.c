#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_var.h>
#include <ppapi/c/ppb_var_array.h>

#include "log.h"
#include "res.h"

#include "PPB_Var.h"

struct PPB_VarArray_1_0 PPB_VarArray_1_0_instance;

typedef struct ppb_var_array_desc
{
    PP_Resource self;
} ppb_var_array_t;

static void ppb_var_array_destructor(ppb_var_array_t* ctx)
{
    LOG_D("{%d}", ctx->self);
};

/**
 * Creates an array var, i.e., a <code>PP_Var</code> with type set to
 * <code>PP_VARTYPE_ARRAY</code>. The array length is set to 0.
 *
 * @return An empty array var, whose reference count is set to 1 on behalf of
 * the caller.
 */
static struct PP_Var Create(void)
{
    struct PP_Var var;
    ppb_var_array_t* dst;

    var.type = PP_VARTYPE_ARRAY;
    var.value.as_id = res_create(sizeof(ppb_var_array_t), &PPB_VarArray_1_0_instance, (res_destructor_t)ppb_var_array_destructor);

    LOG_E("var.value.as_id=%d", (int)var.value.as_id);

    dst = (ppb_var_array_t*)res_private(var.value.as_id);
    dst->self = var.value.as_id;

    return var;
};

/**
 * Gets an element from the array.
 *
 * @param[in] array An array var.
 * @param[in] index An index indicating which element to return.
 *
 * @return The element at the specified position. The reference count of the
 * element returned is incremented on behalf of the caller. If
 * <code>index</code> is larger than or equal to the array length, an
 * undefined var is returned.
 */
static struct PP_Var Get(struct PP_Var array, uint32_t index)
{
    LOG_NP;
    return PP_MakeNull(); //PP_MakeUndefined();
};

/**
 * Sets the value of an element in the array.
 *
 * @param[in] array An array var.
 * @param[in] index An index indicating which element to modify. If
 * <code>index</code> is larger than or equal to the array length, the length
 * is updated to be <code>index</code> + 1. Any position in the array that
 * hasn't been set before is set to undefined, i.e., <code>PP_Var</code> of
 * type <code>PP_VARTYPE_UNDEFINED</code>.
 * @param[in] value The value to set. The array holds a reference to it on
 * success.
 *
 * @return A <code>PP_Bool</code> indicating whether the operation succeeds.
 */
static PP_Bool Set(struct PP_Var array, uint32_t index, struct PP_Var value)
{
    LOG_NP;
    return 0;
};

/**
 * Gets the array length.
 *
 * @param[in] array An array var.
 *
 * @return The array length.
 */
static uint32_t GetLength(struct PP_Var array)
{
    LOG_NP;
    return 0;
};

/**
 * Sets the array length.
 *
 * @param[in] array An array var.
 * @param[in] length The new array length. If <code>length</code> is smaller
 * than its current value, the array is truncated to the new length; any
 * elements that no longer fit are removed and the references to them will be
 * released. If <code>length</code> is larger than its current value,
 * undefined vars are appended to increase the array to the specified length.
 *
 * @return A <code>PP_Bool</code> indicating whether the operation succeeds.
 */
static PP_Bool SetLength(struct PP_Var array, uint32_t length)
{
    LOG_NP;
    return 0;
};


/**
 * @addtogroup Interfaces
 * @{
 */
struct PPB_VarArray_1_0 PPB_VarArray_1_0_instance =
{
    .Create = Create,
    .Get = Get,
    .Set = Set,
    .GetLength = GetLength,
    .SetLength = SetLength,
};
