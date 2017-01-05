#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/private/ppb_flash.h>

#include "PPB_Var.h"
#include "log.h"


static void SetInstanceAlwaysOnTop(PP_Instance instance, PP_Bool on_top)
{
    LOG_TD;
};

static PP_Bool DrawGlyphs(PP_Instance instance,
    PP_Resource pp_image_data,
    const struct PP_BrowserFont_Trusted_Description* font_desc,
    uint32_t color,
    const struct PP_Point* position,
    const struct PP_Rect* clip,
    const float transformation[3][3],
    PP_Bool allow_subpixel_aa,
    uint32_t glyph_count,
    const uint16_t glyph_indices[],
    const struct PP_Point glyph_advances[])
{
    LOG_NP;
    return 0;
};


static struct PP_Var GetProxyForURL(PP_Instance instance, const char* url)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static int32_t Navigate(PP_Resource request_info, const char* target, PP_Bool from_user_action)
{
    LOG_NP;
    return 0;
};

/**
 * Retrieves the local time zone offset from GM time for the given UTC time.
 */
static double GetLocalTimeZoneOffset(PP_Instance instance, PP_Time _t)
{
    struct tm tm;
    time_t t = _t;

    localtime_r(&t, &tm);

    return tm.tm_gmtoff;
};

static struct PP_Var GetCommandLineArgs(PP_Module module)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static void PreloadFontWin(const void* logfontw)
{
    LOG_NP;
};

static PP_Bool IsRectTopmost(PP_Instance instance, const struct PP_Rect* rect)
{
    LOG_NP;
    return 0;
};

static void UpdateActivity(PP_Instance instance)
{
    LOG_NP;
};

#define LOG_GetSetting_case(S) case S: LOG_N("%s", #S);
static struct PP_Var GetSetting(PP_Instance instance, PP_FlashSetting setting)
{
    struct PP_Var r = PP_MakeUndefined();

    switch(setting)
    {
        LOG_GetSetting_case(PP_FLASHSETTING_3DENABLED);
            r = PP_MakeBool(PP_TRUE);
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_INCOGNITO);
            r = PP_MakeBool(PP_FALSE);
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_STAGE3DENABLED);
            r = PP_MakeBool(PP_TRUE);
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_LANGUAGE);
            r =  VarFromUtf8_c("en-US");
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_NUMCORES);
            r = PP_MakeInt32(sysconf(_SC_NPROCESSORS_ONLN));
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_LSORESTRICTIONS);
            r = PP_MakeInt32(PP_FLASHLSORESTRICTIONS_NONE);
            break;

        LOG_GetSetting_case(PP_FLASHSETTING_STAGE3DBASELINEENABLED);
            r = PP_MakeBool(PP_TRUE);
            break;
    };

    return r;
};

static PP_Bool SetCrashData(PP_Instance instance, PP_FlashCrashKey key, struct PP_Var value)
{
    LOG_NP;
    return 0;
};

static int32_t EnumerateVideoCaptureDevices(PP_Instance instance,
    PP_Resource video_capture, struct PP_ArrayOutput devices)
{
    LOG_NP;
    return 0;
};


struct PPB_Flash_13_0 PPB_Flash_13_0_instance =
{
    .SetInstanceAlwaysOnTop = SetInstanceAlwaysOnTop,
    .DrawGlyphs = DrawGlyphs,
    .GetProxyForURL = GetProxyForURL,
    .Navigate = Navigate,
    .GetLocalTimeZoneOffset = GetLocalTimeZoneOffset,
    .GetCommandLineArgs = GetCommandLineArgs,
    .PreloadFontWin = PreloadFontWin,
    .IsRectTopmost = IsRectTopmost,
    .UpdateActivity = UpdateActivity,
    .GetSetting = GetSetting,
    .SetCrashData = SetCrashData,
    .EnumerateVideoCaptureDevices = EnumerateVideoCaptureDevices,
};

static void RunMessageLoop(PP_Instance instance)
{
    LOG_NP;
};

static void QuitMessageLoop(PP_Instance instance)
{
    LOG_NP;
};

static int32_t InvokePrinting(PP_Instance instance)
{
    LOG_NP;
    return 0;
};

static struct PP_Var GetDeviceID(PP_Instance instance)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static int32_t GetSettingInt(PP_Instance instance, PP_FlashSetting setting)
{
    LOG_NP;
    return 0;
};

struct PPB_Flash_12_6 PPB_Flash_12_6_instance =
{
    .SetInstanceAlwaysOnTop = SetInstanceAlwaysOnTop,
    .GetProxyForURL = GetProxyForURL,
    .Navigate = Navigate,
    .RunMessageLoop = RunMessageLoop,
    .QuitMessageLoop = QuitMessageLoop,
    .GetLocalTimeZoneOffset = GetLocalTimeZoneOffset,
    .GetCommandLineArgs = GetCommandLineArgs,
    .PreloadFontWin = PreloadFontWin,
    .IsRectTopmost = IsRectTopmost,
    .InvokePrinting = InvokePrinting,
    .UpdateActivity = UpdateActivity,
    .GetDeviceID = GetDeviceID,
    .GetSettingInt = GetSettingInt,
    .GetSetting = GetSetting,
    .SetCrashData = SetCrashData,
    .EnumerateVideoCaptureDevices = EnumerateVideoCaptureDevices,
};

