#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/ppb_var.h>
#include <ppapi/c/ppb_var_dictionary.h>

#include "log.h"
#include "res.h"

#include "PPB_Var.h"

struct PPB_VarDictionary_1_0 PPB_VarDictionary_1_0_instance;

typedef struct ppb_var_dictionary_desc
{
    PP_Resource self;
} ppb_var_dictionary_t;

static void ppb_var_dictionary_destructor(ppb_var_dictionary_t* ctx)
{
    LOG_D("{%d}", ctx->self);
};


  /**
   * Creates a dictionary var, i.e., a <code>PP_Var</code> with type set to
   * <code>PP_VARTYPE_DICTIONARY</code>.
   *
   * @return An empty dictionary var, whose reference count is set to 1 on
   * behalf of the caller.
   */
static struct PP_Var Create(void)
{
    struct PP_Var var;
    ppb_var_dictionary_t* dst;

    var.type = PP_VARTYPE_ARRAY_BUFFER;
    var.value.as_id = res_create(sizeof(ppb_var_dictionary_t), &PPB_VarDictionary_1_0_instance, (res_destructor_t)ppb_var_dictionary_destructor);

    LOG_E("var.value.as_id=%d", (int)var.value.as_id);

    dst = (ppb_var_dictionary_t*)res_private(var.value.as_id);
    dst->self = var.value.as_id;

    return var;
};

  /**
   * Gets the value associated with the specified key.
   *
   * @param[in] dict A dictionary var.
   * @param[in] key A string var.
   *
   * @return The value that is associated with <code>key</code>. The reference
   * count of the element returned is incremented on behalf of the caller. If
   * <code>key</code> is not a string var, or it doesn't exist in
   * <code>dict</code>, an undefined var is returned.
   */
static struct PP_Var Get(struct PP_Var dict, struct PP_Var key)
{
    LOG_NP;
    return PP_MakeUndefined();
};

  /**
   * Sets the value associated with the specified key.
   *
   * @param[in] dict A dictionary var.
   * @param[in] key A string var. If this key hasn't existed in
   * <code>dict</code>, it is added and associated with <code>value</code>;
   * otherwise, the previous value is replaced with <code>value</code>.
   * @param[in] value The value to set. The dictionary holds a reference to it
   * on success.
   *
   * @return A <code>PP_Bool</code> indicating whether the operation succeeds.
   */
static PP_Bool Set(struct PP_Var dict, struct PP_Var key, struct PP_Var value)
{
    LOG_NP;
    return 0;
};

  /**
   * Deletes the specified key and its associated value, if the key exists. The
   * reference to the element will be released.
   *
   * @param[in] dict A dictionary var.
   * @param[in] key A string var.
   */
static void Delete(struct PP_Var dict, struct PP_Var key)
{
    LOG_NP;
};

  /**
   * Checks whether a key exists.
   *
   * @param[in] dict A dictionary var.
   * @param[in] key A string var.
   *
   * @return A <code>PP_Bool</code> indicating whether the key exists.
   */
static PP_Bool HasKey(struct PP_Var dict, struct PP_Var key)
{
    LOG_NP;
    return 0;
};

  /**
   * Gets all the keys in a dictionary. Please note that for each key that you
   * set into the dictionary, a string var with the same contents is returned;
   * but it may not be the same string var (i.e., <code>value.as_id</code> may
   * be different).
   *
   * @param[in] dict A dictionary var.
   *
   * @return An array var which contains all the keys of <code>dict</code>. Its
   * reference count is incremented on behalf of the caller. The elements are
   * string vars. Returns a null var if failed.
   */
static struct PP_Var GetKeys(struct PP_Var dict)
{
    LOG_NP;
    return PP_MakeUndefined();
};


/**
 * @addtogroup Interfaces
 * @{
 */
/**
 * A dictionary var contains key-value pairs with unique keys. The keys are
 * strings while the values can be arbitrary vars. Key comparison is always
 * done by value instead of by reference.
 */
struct PPB_VarDictionary_1_0 PPB_VarDictionary_1_0_instance =
{
    .Create = Create,
    .Get = Get,
    .Set = Set,
    .Delete = Delete,
    .HasKey = HasKey,
    .GetKeys = GetKeys,
};
