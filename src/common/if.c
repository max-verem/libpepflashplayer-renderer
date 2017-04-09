#include <stdio.h>
#include <stdlib.h>

#include <string.h>

#include "log.h"

#include "if.h"

#include "PPB.h"

const static if_t ifs[] =
{
    {
        .name = PPB_CORE_INTERFACE_1_0,
        .ptr = &PPB_Core_1_0_interface,
    },
    {
        .name = PPB_AUDIO_CONFIG_INTERFACE_1_1,
        .ptr = &PPB_AudioConfig_1_1_instance,
    },
    {
        .name = PPB_AUDIO_INPUT_DEV_INTERFACE_0_4,
        .ptr = &PPB_AudioInput_Dev_0_4_instance,
    },
    {
        .name = PPB_AUDIO_INTERFACE_1_1,
        .ptr = &PPB_Audio_1_1_instance,
    },
    {
        .name = PPB_BROWSERFONT_TRUSTED_INTERFACE_1_0,
        .ptr = &PPB_BrowserFont_Trusted_1_0_instance,
    },
    {
        .name = PPB_BUFFER_DEV_INTERFACE_0_4,
        .ptr = &PPB_Buffer_Dev_0_4_instance,
    },
    {
        .name = PPB_CHAR_SET_DEV_INTERFACE_0_4,
        .ptr = &PPB_CharSet_Dev_0_4_instance,
    },
    {
        .name = PPB_CRYPTO_DEV_INTERFACE_0_1,
        .ptr = &PPB_Crypto_Dev_0_1_instance,
    },
    {
        .name = PPB_CURSOR_CONTROL_DEV_INTERFACE_0_4,
        .ptr = &PPB_CursorControl_Dev_0_4_instance,
    },
    {
        .name = PPB_FILECHOOSER_DEV_INTERFACE_0_6,
        .ptr = &PPB_FileChooser_Dev_0_6_instance,
    },
    {
        .name = PPB_FILECHOOSER_TRUSTED_INTERFACE_0_6,
        .ptr = &PPB_FileChooserTrusted_0_6_instance,
    },
    {
        .name = PPB_FILEREF_INTERFACE_1_2,
        .ptr = &PPB_FileRef_1_2_instance,
    },
    {
        .name = PPB_FLASH_CLIPBOARD_INTERFACE_5_1,
        .ptr = &PPB_Flash_Clipboard_5_1_instance,
    },
    {
        .name = PPB_FLASH_FILE_FILEREF_INTERFACE,
        .ptr = &PPB_Flash_File_FileRef_instance,
    },
    {
        .name = PPB_FLASH_FILE_MODULELOCAL_INTERFACE_3_0,
        .ptr = &PPB_Flash_File_ModuleLocal_3_0_instance,
    },
    {
        .name = PPB_FLASH_FONTFILE_INTERFACE_0_2,
        .ptr = &PPB_Flash_FontFile_0_2_instance,
    },
    {
        .name = PPB_FLASHFULLSCREEN_INTERFACE_1_0,
        .ptr = &PB_FlashFullscreen_1_0_instance,
    },
    {
        .name = PPB_FLASH_INTERFACE_13_0,
        .ptr = &PPB_Flash_13_0_instance,
    },
    {
        .name = PPB_FLASH_MENU_INTERFACE_0_2,
        .ptr = &PPB_Flash_Menu_0_2_instance,
    },
    {
        .name = PPB_GRAPHICS_2D_INTERFACE_1_1,
        .ptr = &PPB_Graphics2D_1_1_instance,
    },
    {
        .name = PPB_GRAPHICS_3D_INTERFACE_1_0,
        .ptr = &PPB_Graphics3D_1_0_instance,
    },
    {
        .name = PPB_IMAGEDATA_INTERFACE_1_0,
        .ptr = &PPB_ImageData_1_0_instance,
    },
    {
        .name = PPB_IME_INPUT_EVENT_DEV_INTERFACE_0_2,
        .ptr = &PPB_IMEInputEvent_Dev_0_2_instance,
    },
    {
        .name = PPB_INPUT_EVENT_INTERFACE_1_0,
        .ptr = &PPB_InputEvent_1_0_instance,
    },
    {
        .name = PPB_INSTANCE_INTERFACE_1_0,
        .ptr = &PPB_Instance_1_0_instance,
    },
    {
        .name = PPB_MEMORY_DEV_INTERFACE_0_1,
        .ptr = &PPB_Memory_Dev_0_1_instance,
    },
    {
        .name = PPB_NETADDRESS_PRIVATE_INTERFACE_1_1,
        .ptr = &PPB_NetAddress_Private_1_1_instance,
    },
    {
        .name = PPB_OPENGLES2_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_CHROMIUMENABLEFEATURE_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2ChromiumEnableFeature_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_CHROMIUMMAPSUB_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2ChromiumMapSub_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_DRAWBUFFERS_DEV_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2DrawBuffers_Dev_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_FRAMEBUFFERBLIT_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2FramebufferBlit_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_FRAMEBUFFERMULTISAMPLE_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2FramebufferMultisample_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_INSTANCEDARRAYS_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2InstancedArrays_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_QUERY_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2Query_1_0_instance,
    },
    {
        .name = PPB_OPENGLES2_VERTEXARRAYOBJECT_INTERFACE_1_0,
        .ptr = &PPB_OpenGLES2VertexArrayObject_1_0_instance,
    },
    {
        .name = PPB_TCPSOCKET_PRIVATE_INTERFACE_0_5,
        .ptr = &PPB_TCPSocket_Private_0_5_instance,
    },
    {
        .name = PPB_TEXTINPUT_DEV_INTERFACE_0_2,
        .ptr = &PPB_TextInput_Dev_0_2_instance,
    },
    {
        .name = PPB_UDPSOCKET_PRIVATE_INTERFACE_0_4,
        .ptr = &PPB_UDPSocket_Private_0_4_instance,
    },
    {
        .name = PPB_URLLOADER_INTERFACE_1_0,
        .ptr = &PPB_URLLoader_1_0_instance,
    },
    {
        .name = PPB_URLLOADERTRUSTED_INTERFACE_0_3,
        .ptr = &PPB_URLLoaderTrusted_0_3_instance,
    },
    {
        .name = PPB_URLREQUESTINFO_INTERFACE_1_0,
        .ptr = &PPB_URLRequestInfo_1_0_instance,
    },
    {
        .name = PPB_URLRESPONSEINFO_INTERFACE_1_0,
        .ptr = &PPB_URLResponseInfo_1_0_instance,
    },
    {
        .name = PPB_URLUTIL_DEV_INTERFACE_0_7,
        .ptr = &PPB_URLUtil_Dev_0_7_instance,
    },
    {
        .name = PPB_VAR_INTERFACE_1_2,
        .ptr = &PPB_Var_1_2_instance,
    },
    {
        .name = PPB_VIDEOCAPTURE_DEV_INTERFACE_0_3,
        .ptr = &PPB_VideoCapture_Dev_0_3_instance,
    },
    {
        .name = PPB_VIEW_INTERFACE_1_2,
        .ptr = &PPB_View_1_2_instance,
    },
    {
        .name = PPB_PRINTING_DEV_INTERFACE_0_7,
        .ptr = &PPB_Printing_Dev_0_7_instance,
    },
    {
        .name = PPB_VAR_DEPRECATED_INTERFACE_0_3,
        .ptr = &PPB_Var_Deprecated_instance,
    },
    {
        .name = PPB_BROKER_TRUSTED_INTERFACE_0_3,
        .ptr = &PPB_BrokerTrusted_0_3_instance,
    },
    {
        .name = PPB_NETWORKMONITOR_INTERFACE_1_0,
        .ptr = &PPB_NetworkMonitor_1_0_instance,
    },
    {
        .name = PPB_MESSAGELOOP_INTERFACE_1_0,
        .ptr = &PPB_MessageLoop_1_0_instance,
    },
    {
        .name = PPB_FLASH_INTERFACE_12_6,
        .ptr = &PPB_Flash_12_6_instance,
    },
    {
        .name = PPB_INSTANCE_PRIVATE_INTERFACE_0_1,
        .ptr = &PPB_Instance_Private_0_1_instance,
    },
    {
        .name = PPB_VAR_ARRAY_BUFFER_INTERFACE_1_0,
        .ptr = &PPB_VarArrayBuffer_1_0_instance,
    },
    {
        .name = PPB_VAR_ARRAY_INTERFACE_1_0,
        .ptr = &PPB_VarArray_1_0_instance,
    },
    {
        .name = PPB_VAR_DICTIONARY_INTERFACE_1_0,
        .ptr = &PPB_VarDictionary_1_0_instance,
    },
    {
        .name = PPB_NETADDRESS_INTERFACE_1_0,
        .ptr = &PPB_NetAddress_1_0_instance,
    },
    {
        .name = NULL,
        .ptr = NULL,
    }
};

const if_t* if_find(const char* name)
{
    int i;

    for(i = 0; ifs[i].ptr; i++)
        if(!strcmp(name, ifs[i].name))
            return &ifs[i];

    return NULL;
};

const void* get_browser_interface_proc(const char* interface_name)
{
    const if_t* i;

    LOG_T("interface_name=[%s]", interface_name);

    i = if_find(interface_name);

    if(i)
    {
        LOG_T("FOUND=[%s]", interface_name);
        return i->ptr;
    }

    LOG_E("ABSENT=[%s]", interface_name);

    return NULL;
};
