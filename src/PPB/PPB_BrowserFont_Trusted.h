#ifndef PPB_BrowserFont_Trusted_h
#define PPB_BrowserFont_Trusted_h

#include <ppapi/c/pp_var.h>
#include <ppapi/c/pp_instance.h>
#include <ppapi/c/trusted/ppb_browser_font_trusted.h>

#include <pango/pangoft2.h>

typedef struct browser_font_trusted_desc
{
    PP_Instance instance_id;
    PP_Resource self;

    struct PP_BrowserFont_Trusted_Description description;

    /*PangoFT2FontMap*/PangoFontMap *fontmap;
    PangoContext *context;
//    PangoLayout *layout;
    PangoFont *font;
} browser_font_trusted_t;

#endif /* PPB_BrowserFont_Trusted_h */
