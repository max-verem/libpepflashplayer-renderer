#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/trusted/ppb_browser_font_trusted.h>

#include "log.h"

static struct PP_Var GetFontFamilies(PP_Instance instance)
{
    LOG_NP;
    return PP_MakeInt32(0);
};

static PP_Resource Create(PP_Instance instance, const struct PP_BrowserFont_Trusted_Description* description)
{
    LOG_NP;
    return 0;
};

static PP_Bool IsFont(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

static PP_Bool Describe(PP_Resource font,
    struct PP_BrowserFont_Trusted_Description* description,
    struct PP_BrowserFont_Trusted_Metrics* metrics)
{
    LOG_NP;
    return 0;
};

static PP_Bool DrawTextAt(PP_Resource font, PP_Resource image_data,
    const struct PP_BrowserFont_Trusted_TextRun* text, const struct PP_Point* position,
    uint32_t color, const struct PP_Rect* clip, PP_Bool image_data_is_opaque)
{
    LOG_NP;
    return 0;
};

static int32_t MeasureText(PP_Resource font, const struct PP_BrowserFont_Trusted_TextRun* text)
{
    LOG_NP;
    return 0;
};

static uint32_t CharacterOffsetForPixel(PP_Resource font,
    const struct PP_BrowserFont_Trusted_TextRun* text, int32_t pixel_position)
{
    LOG_NP;
    return 0;
};

static int32_t PixelOffsetForCharacter(PP_Resource font,
    const struct PP_BrowserFont_Trusted_TextRun* text, uint32_t char_offset)
{
    LOG_NP;
    return 0;
};

struct PPB_BrowserFont_Trusted_1_0 PPB_BrowserFont_Trusted_1_0_instance =
{
    .GetFontFamilies = GetFontFamilies,
    .Create = Create,
    .IsFont = IsFont,
    .Describe = Describe,
    .DrawTextAt = DrawTextAt,
    .MeasureText = MeasureText,
    .CharacterOffsetForPixel = CharacterOffsetForPixel,
    .PixelOffsetForCharacter = PixelOffsetForCharacter,
};
