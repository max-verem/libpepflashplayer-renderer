#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash_font_file.h>

#include "log.h"

static PP_Resource Create(PP_Instance instance,
    const struct PP_BrowserFont_Trusted_Description* description,
    PP_PrivateFontCharset charset)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFlashFontFile(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static PP_Bool GetFontTable(PP_Resource font_file,
    uint32_t table, void* output, uint32_t* output_length)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsSupportedForWindows(void)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_FontFile_0_2 PPB_Flash_FontFile_0_2_instance =
{
    .Create = Create,
    .IsFlashFontFile = IsFlashFontFile,
    .GetFontTable = GetFontTable,
    .IsSupportedForWindows = IsSupportedForWindows
};
