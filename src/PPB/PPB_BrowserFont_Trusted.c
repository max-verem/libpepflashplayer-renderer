#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/trusted/ppb_browser_font_trusted.h>

#include "log.h"

#include <glib.h>
#include <pango/pangoft2.h>

#include "PPB_Var.h"

static int pango_font_family_cmp_name(PangoFontFamily* fa, PangoFontFamily* fb)
{
    const char* na = pango_font_family_get_name(fa);
    const char* nb = pango_font_family_get_name(fb);
    return strcmp(na, nb);
};

/**
 * Returns a list of all available font families on the system. You can use
 * this list to decide whether to Create() a font.
 *
 * The return value will be a single string with null characters delimiting
 * the end of each font name. For example: "Arial\0Courier\0Times\0".
 *
 * Returns an undefined var on failure (this typically means you passed an
 * invalid instance).
 */
static struct PP_Var GetFontFamilies(PP_Instance instance)
{
    struct PP_Var r;
    PangoFontMap *fontmap;
    PangoFontFamily **families;
    int i, j, n_families;

    char* buf_data = (char*)malloc(1);
    int buf_size = 0;

    *buf_data = 0;

    /* get fontmap */
    fontmap = pango_ft2_font_map_new();

    /* get font families list */
    pango_font_map_list_families(fontmap, &families, &n_families);

    /* sort */
    for(i = 0; i < n_families; i++)
        for(j = i + 1; j < n_families; j++)
            if(pango_font_family_cmp_name(families[i], families[j]) > 0)
            {
                PangoFontFamily *f = families[i];
                families[i] = families[j];
                families[j] = f;
            };

    /* build a string */
    for(i = 0; i < n_families; i++)
    {
        int l;
        const char * family_name;
        PangoFontFamily * family = families[i];

        family_name = pango_font_family_get_name (family);

        l = strlen(family_name);

        buf_data = (char*)realloc(buf_data, buf_size + l + 1);
        memcpy(buf_data + buf_size, family_name, l + 1);
        buf_size += l + 1;
    };

    g_free (families);

    g_object_unref(fontmap);

    r = VarFromUtf8(buf_data, buf_size);

    free(buf_data);

    return r;
};

/**
 * Returns a font which best matches the given description. The return value
 * will have a non-zero ID on success, or zero on failure.
 */
static PP_Resource Create(PP_Instance instance, const struct PP_BrowserFont_Trusted_Description* description)
{
    LOG_NP;
    return 0;
};

/**
 * Returns PP_TRUE if the given resource is a Font. Returns PP_FALSE if the
 * resource is invalid or some type other than a Font.
 */
static PP_Bool IsFont(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

/**
 * Loads the description and metrics of the font into the given structures.
 * The description will be different than the description the font was
 * created with since it will be filled with the real values from the font
 * that was actually selected.
 *
 * The PP_Var in the description should be of type Void on input. On output,
 * this will contain the string and will have a reference count of 1. The
 * plugin is responsible for calling Release on this var.
 *
 * Returns PP_TRUE on success, PP_FALSE if the font is invalid or if the Var
 * in the description isn't Null (to prevent leaks).
 */
static PP_Bool Describe(PP_Resource font,
    struct PP_BrowserFont_Trusted_Description* description,
    struct PP_BrowserFont_Trusted_Metrics* metrics)
{
    LOG_NP;
    return 0;
};

/**
 * Draws the text to the image buffer.
 *
 * The given point represents the baseline of the left edge of the font,
 * regardless of whether it is left-to-right or right-to-left (in the case of
 * RTL text, this will actually represent the logical end of the text).
 *
 * The clip is optional and may be NULL. In this case, the text will be
 * clipped to the image.
 *
 * The image_data_is_opaque flag indicates whether subpixel antialiasing can
 * be performed, if it is supported. When the image below the text is
 * opaque, subpixel antialiasing is supported and you should set this to
 * PP_TRUE to pick up the user's default preferences. If your plugin is
 * partially transparent, then subpixel antialiasing is not possible and
 * grayscale antialiasing will be used instead (assuming the user has
 * antialiasing enabled at all).
 */
static PP_Bool DrawTextAt(PP_Resource font, PP_Resource image_data,
    const struct PP_BrowserFont_Trusted_TextRun* text, const struct PP_Point* position,
    uint32_t color, const struct PP_Rect* clip, PP_Bool image_data_is_opaque)
{
    LOG_NP;
    return 0;
};

/**
 * Returns the width of the given string. If the font is invalid or the var
 * isn't a valid string, this will return -1.
 *
 * Note that this function handles complex scripts such as Arabic, combining
 * accents, etc. so that adding the width of substrings won't necessarily
 * produce the correct width of the entire string.
 *
 * Returns -1 on failure.
 */
static int32_t MeasureText(PP_Resource font, const struct PP_BrowserFont_Trusted_TextRun* text)
{
    LOG_NP;
    return 0;
};

/**
 * Returns the character at the given pixel X position from the beginning of
 * the string. This handles complex scripts such as Arabic, where characters
 * may be combined or replaced depending on the context. Returns (uint32)-1
 * on failure.
 *
 * TODO(brettw) this function may be broken. See the CharPosRTL test. It
 * seems to tell you "insertion point" rather than painting position. This
 * is useful but maybe not what we intended here.
 */
static uint32_t CharacterOffsetForPixel(PP_Resource font,
    const struct PP_BrowserFont_Trusted_TextRun* text, int32_t pixel_position)
{
    LOG_NP;
    return 0;
};

/**
 * Returns the horizontal advance to the given character if the string was
 * placed at the given position. This handles complex scripts such as Arabic,
 * where characters may be combined or replaced depending on context. Returns
 * -1 on error.
 */
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
