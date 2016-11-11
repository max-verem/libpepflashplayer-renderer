#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_text_input_dev.h>

#include "log.h"

/**
 * Informs the browser about the current text input mode of the plugin.
 * Typical use of this information in the browser is to properly
 * display/suppress tools for supporting text inputs (such as virtual
 * keyboards in touch screen based devices, or input method editors often
 * used for composing East Asian characters).
 */
static void SetTextInputType(PP_Instance instance, PP_TextInput_Type_Dev type)
{
    LOG_NP;
};

/**
 * Informs the browser about the coordinates of the text input caret and the
 * bounding box of the text input area. Typical use of this information in
 * the browser is to layout IME windows etc.
 */
static void UpdateCaretPosition(PP_Instance instance,
    const struct PP_Rect* caret, const struct PP_Rect* bounding_box)
{
    LOG_NP;
};

/**
 * Cancels the current composition in IME.
 */
static void CancelCompositionText(PP_Instance instance)
{
    LOG_NP;
};

/**
 * In response to the <code>PPP_TextInput_Dev::RequestSurroundingText</code>
 * call, informs the browser about the current text selection and surrounding
 * text. <code>text</code> is a UTF-8 string that contains the current range
 * of text selection in the plugin. <code>caret</code> is the byte-index of
 * the caret position within <code>text</code>. <code>anchor</code> is the
 * byte-index of the anchor position (i.e., if a range of text is selected,
 * it is the other edge of selection different from <code>caret</code>. If
 * there are no selection, <code>anchor</code> is equal to <code>caret</code>.
 *
 * Typical use of this information in the browser is to enable "reconversion"
 * features of IME that puts back the already committed text into the
 * pre-commit composition state. Another use is to improve the precision
 * of suggestion of IME by taking the context into account (e.g., if the caret
 * looks to be on the beginning of a sentence, suggest capital letters in a
 * virtual keyboard).
 *
 * When the focus is not on text, call this function setting <code>text</code>
 * to an empty string and <code>caret</code> and <code>anchor</code> to zero.
 * Also, the plugin should send the empty text when it does not want to reveal
 * the selection to IME (e.g., when the surrounding text is containing
 * password text).
 */
static void UpdateSurroundingText(PP_Instance instance,
    const char* text, uint32_t caret, uint32_t anchor)
{
    LOG_NP;
};

static void SelectionChanged(PP_Instance instance)
{
    LOG_NP;
};

struct PPB_TextInput_Dev_0_2 PPB_TextInput_Dev_0_2_instance =
{
    .SetTextInputType = SetTextInputType,
    .UpdateCaretPosition = UpdateCaretPosition,
    .CancelCompositionText = CancelCompositionText,
    .UpdateSurroundingText = UpdateSurroundingText,
    .SelectionChanged = SelectionChanged,
};
