#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash_menu.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance_id, const struct PP_Flash_Menu* menu_data)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFlashMenu(PP_Resource resource_id)
{
    LOG_NP;
    return 0;
};

static int32_t Show(PP_Resource menu_id, const struct PP_Point* location,
    int32_t* selected_id, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_Menu_0_2 PPB_Flash_Menu_0_2_instance =
{
    .Create = Create,
    .IsFlashMenu = IsFlashMenu,
    .Show = Show,
};
