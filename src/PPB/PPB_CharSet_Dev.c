#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <iconv.h>
#include <errno.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_char_set_dev.h>

#include "log.h"
#include "PPB_Var.h"
#include "PPB_Memory_Dev.h"

// Converts the UTF-16 string pointed to in |*utf16| to an 8-bit string in the
// specified code page. |utf16_len| is measured in UTF-16 units, not bytes.
// This value may not be NULL.
//
// The return value is a NULL-terminated 8-bit string corresponding to the
// new character set, or NULL on failure. THIS STRING MUST BE FREED USING
// PPB_Core::MemFree(). The length of the returned string, not including the
// terminating NULL, will be placed into *output_length. When there is no
// error, the result will always be non-NULL, even if the output is 0-length.
// In this case, it will only contain the terminator. You must still call
// MemFree any time the return value is non-NULL.
//
// This function will return NULL if there was an error converting the string
// and you requested PP_CHARSET_CONVERSIONERROR_FAIL, or the output character
// set was unknown.
char* UTF16ToCharSet(PP_Instance instance,
    const uint16_t* utf16, uint32_t utf16_len,
    const char* output_char_set,
    enum PP_CharSet_ConversionError on_error,
    uint32_t* output_length)
{
    size_t r;
    iconv_t i;
    char *buf, *inbuf, *outbuf;
    size_t inbytesleft, outbytesleft;

    LOG_T("{%d}", instance);

    // const char *tocode, const char *fromcode
    i = iconv_open(output_char_set, "UTF-16");
    if(i == (iconv_t) -1)
    {
        LOG_E("iconv_open([%s], UTF-16) failed", output_char_set);
        return NULL;
    };

    // prep
    inbytesleft = 2 * utf16_len;
    inbuf = (char*)utf16;

    outbytesleft = 8 * utf16_len;
    outbuf = buf = (char*)MemAlloc(outbytesleft);

    // conv
    r = iconv(i, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
    if(r == (size_t) -1)
    {
        LOG_E("iconv() failed, errno=%d", errno);
        MemFree(buf);
        buf = NULL;
    }
    else
    {
        *output_length = outbuf - buf;
    };

    iconv_close(i);

    return buf;
};

// Same as UTF16ToCharSet except converts in the other direction. The input
// is in the given charset, and the |input_len| is the number of bytes in
// the |input| string. |*output_length| is the number of 16-bit values in
// the output not counting the terminating NULL.
//
// Since UTF16 can represent every Unicode character, the only time the
// replacement character will be used is if the encoding in the input string
// is incorrect.
static uint16_t* CharSetToUTF16(PP_Instance instance,
    const char* input, uint32_t input_len,
    const char* input_char_set,
    enum PP_CharSet_ConversionError on_error,
    uint32_t* output_length)
{
    size_t r;
    iconv_t i;
    char *buf, *inbuf, *outbuf;
    size_t inbytesleft, outbytesleft;

    LOG_T("{%d}", instance);

    // const char *tocode, const char *fromcode
    i = iconv_open("UTF-16", input_char_set);
    if(i == (iconv_t) -1)
    {
        LOG_E("iconv_open(UTF-16, [%s]) failed", input_char_set);
        return NULL;
    };

    // prep
    inbuf = (char*)input;
    inbytesleft = 2 * input_len;

    outbytesleft = 8 * input_len;
    outbuf = buf = MemAlloc(outbytesleft);

    // conv
    r = iconv(i, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
    if(r == (size_t) -1)
    {
        LOG_E("iconv() failed, errno=%d", errno);
        MemFree(buf);
        buf = NULL;
    }
    else
    {
        *output_length = outbuf - buf;
    };

    iconv_close(i);

    return (uint16_t*)buf;
};

static struct PP_Var GetDefaultCharSet(PP_Instance instance)
{
    return VarFromUtf8_c("UTF-8");
};

struct PPB_CharSet_Dev_0_4 PPB_CharSet_Dev_0_4_instance =
{
    .UTF16ToCharSet = UTF16ToCharSet,
    .CharSetToUTF16 = CharSetToUTF16,
    .GetDefaultCharSet = GetDefaultCharSet,
};
