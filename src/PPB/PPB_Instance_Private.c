#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_instance_private.h>

#include "log.h"

#include "PPB_Var.h"

/**
 * GetWindowObject is a pointer to a function that determines
 * the DOM window containing this module instance.
 *
 * @param[in] instance A PP_Instance whose WindowObject should be retrieved.
 * @return A PP_Var containing window object on success.
 */
static struct PP_Var GetWindowObject(PP_Instance instance)
{
    LOG_TD;
    return VarFromResource(instance);
};

/**
 * GetOwnerElementObject is a pointer to a function that determines
 * the DOM element containing this module instance.
 *
 * @param[in] instance A PP_Instance whose WindowObject should be retrieved.
 * @return A PP_Var containing DOM element on success.
 */
static struct PP_Var GetOwnerElementObject(PP_Instance instance)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

/**
 * ExecuteScript is a pointer to a function that executes the given
 * script in the context of the frame containing the module.
 *
 * The exception, if any, will be returned in *exception. As with the PPB_Var
 * interface, the exception parameter, if non-NULL, must be initialized
 * to a "void" var or the function will immediately return. On success,
 * the exception parameter will be set to a "void" var. On failure, the
 * return value will be a "void" var.
 *
 * @param[in] script A string containing the JavaScript to execute.
 * @param[in/out] exception PP_Var containing the exception. Initialize
 * this to NULL if you don't want exception info; initialize this to a void
 * exception if want exception info.
 *
 * @return The result of the script execution, or a "void" var
 * if execution failed.
 */
static struct PP_Var ExecuteScript(PP_Instance instance, struct PP_Var script, struct PP_Var* exception)
{
    LOG_NP;
    return PP_MakeInt32(0);
};


/**
 * The PPB_Instance_Private interface contains functions available only to
 * trusted plugin instances.
 *
 */
struct PPB_Instance_Private_0_1 PPB_Instance_Private_0_1_instance =
{
    .GetWindowObject = GetWindowObject,
    .GetOwnerElementObject = GetOwnerElementObject,
    .ExecuteScript = ExecuteScript,
};
