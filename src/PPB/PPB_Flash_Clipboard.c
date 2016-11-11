#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash_clipboard.h>

#include "log.h"

static uint32_t RegisterCustomFormat(PP_Instance instance_id, const char* format_name)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFormatAvailable(PP_Instance instance_id,
    PP_Flash_Clipboard_Type clipboard_type, uint32_t format)
{
    LOG_NP;
    return 0;
};

static struct PP_Var ReadData(PP_Instance instance_id,
    PP_Flash_Clipboard_Type clipboard_type, uint32_t format)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static int32_t WriteData(PP_Instance instance_id,
    PP_Flash_Clipboard_Type clipboard_type, uint32_t data_item_count,
    const uint32_t formats[], const struct PP_Var data_items[])
{
    LOG_NP;
    return 0;
};

static PP_Bool GetSequenceNumber(PP_Instance instance_id,
    PP_Flash_Clipboard_Type clipboard_type, uint64_t* sequence_number)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_Clipboard_5_1 PPB_Flash_Clipboard_5_1_instance =
{
    .RegisterCustomFormat = RegisterCustomFormat,
    .IsFormatAvailable = IsFormatAvailable,
    .ReadData = ReadData,
    .WriteData = WriteData,
    .GetSequenceNumber = GetSequenceNumber,
};
