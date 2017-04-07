#include <ppapi/c/ppb_core.h>
#include <ppapi/c/ppb_audio_config.h>
#include <ppapi/c/dev/ppb_audio_input_dev.h>
#include <ppapi/c/ppb_audio.h>
#include <ppapi/c/trusted/ppb_browser_font_trusted.h>
#include <ppapi/c/dev/ppb_buffer_dev.h>
#include <ppapi/c/dev/ppb_char_set_dev.h>
#include <ppapi/c/dev/ppb_crypto_dev.h>
#include <ppapi/c/dev/ppb_cursor_control_dev.h>
#include <ppapi/c/dev/ppb_file_chooser_dev.h>
#include <ppapi/c/trusted/ppb_file_chooser_trusted.h>
#include <ppapi/c/ppb_file_ref.h>
#include <ppapi/c/private/ppb_flash_clipboard.h>
#include <ppapi/c/dev/deprecated_bool.h>
#include <ppapi/c/private/ppb_flash_file.h>
#include <ppapi/c/private/ppb_flash_font_file.h>
#include <ppapi/c/private/ppb_flash_fullscreen.h>
#include <ppapi/c/private/ppb_flash.h>
#include <ppapi/c/private/ppb_flash_menu.h>
#include <ppapi/c/ppb_graphics_2d.h>
#include <ppapi/c/ppb_graphics_3d.h>
#include <ppapi/c/ppb_image_data.h>
#include <ppapi/c/private/ppb_ime_input_event_dev.h>
#include <ppapi/c/ppb_input_event.h>
#include <ppapi/c/ppb_instance.h>
#include <ppapi/c/dev/ppb_memory_dev.h>
#include <ppapi/c/private/ppb_net_address_private.h>
#include <ppapi/c/ppb_opengles2.h>
#include <ppapi/c/dev/ppb_opengles2ext_dev.h>
#include <ppapi/c/private/ppb_tcp_socket_private.h>
#include <ppapi/c/dev/ppb_text_input_dev.h>
#include <ppapi/c/private/ppb_udp_socket_private.h>
#include <ppapi/c/ppb_url_loader.h>
#include <ppapi/c/trusted/ppb_url_loader_trusted.h>
#include <ppapi/c/ppb_url_request_info.h>
#include <ppapi/c/ppb_url_response_info.h>
#include <ppapi/c/dev/ppb_url_util_dev.h>
#include <ppapi/c/ppb_var.h>
#include <ppapi/c/dev/ppb_video_capture_dev.h>
#include <ppapi/c/ppb_view.h>
#include <ppapi/c/dev/ppb_printing_dev.h>
#include <ppapi/c/dev/ppb_var_deprecated.h>
#include <ppapi/c/trusted/ppb_broker_trusted.h>
#include <ppapi/c/ppb_network_monitor.h>
#include <ppapi/c/ppb_message_loop.h>
#include <ppapi/c/private/ppb_instance_private.h>
#include <ppapi/c/ppb_var_array.h>
#include <ppapi/c/ppb_var_array_buffer.h>
#include <ppapi/c/ppb_var_dictionary.h>

extern struct PPB_Core_1_0 PPB_Core_1_0_interface;
extern struct PPB_AudioConfig_1_1 PPB_AudioConfig_1_1_instance;
extern struct PPB_AudioInput_Dev_0_4 PPB_AudioInput_Dev_0_4_instance;
extern struct PPB_Audio_1_1 PPB_Audio_1_1_instance;
extern struct PPB_BrowserFont_Trusted_1_0 PPB_BrowserFont_Trusted_1_0_instance;
extern struct PPB_Buffer_Dev_0_4 PPB_Buffer_Dev_0_4_instance;
extern struct PPB_CharSet_Dev_0_4 PPB_CharSet_Dev_0_4_instance;
extern struct PPB_Crypto_Dev_0_1 PPB_Crypto_Dev_0_1_instance;
extern struct PPB_CursorControl_Dev_0_4 PPB_CursorControl_Dev_0_4_instance;
extern struct PPB_FileChooser_Dev_0_6 PPB_FileChooser_Dev_0_6_instance;
extern struct PPB_FileChooserTrusted_0_6 PPB_FileChooserTrusted_0_6_instance;
extern struct PPB_FileRef_1_2 PPB_FileRef_1_2_instance;
extern struct PPB_Flash_Clipboard_5_1 PPB_Flash_Clipboard_5_1_instance;
extern struct PPB_Flash_File_FileRef PPB_Flash_File_FileRef_instance;
extern struct PPB_Flash_File_ModuleLocal_3_0 PPB_Flash_File_ModuleLocal_3_0_instance;
extern struct PPB_Flash_FontFile_0_2 PPB_Flash_FontFile_0_2_instance;
extern struct PPB_FlashFullscreen_1_0 PB_FlashFullscreen_1_0_instance;
extern struct PPB_Flash_13_0 PPB_Flash_13_0_instance;
extern struct PPB_Flash_Menu_0_2 PPB_Flash_Menu_0_2_instance;
extern struct PPB_Graphics2D_1_1 PPB_Graphics2D_1_1_instance;
extern struct PPB_Graphics3D_1_0 PPB_Graphics3D_1_0_instance;
extern struct PPB_ImageData_1_0 PPB_ImageData_1_0_instance;
extern struct PPB_IMEInputEvent_Dev_0_2 PPB_IMEInputEvent_Dev_0_2_instance;
extern struct PPB_InputEvent_1_0 PPB_InputEvent_1_0_instance;
extern struct PPB_Instance_1_0 PPB_Instance_1_0_instance;
extern struct PPB_Memory_Dev_0_1 PPB_Memory_Dev_0_1_instance;
extern struct PPB_NetAddress_Private_1_1 PPB_NetAddress_Private_1_1_instance;
extern struct PPB_OpenGLES2_1_0 PPB_OpenGLES2_1_0_instance;
extern struct PPB_OpenGLES2ChromiumEnableFeature_1_0 PPB_OpenGLES2ChromiumEnableFeature_1_0_instance;
extern struct PPB_OpenGLES2ChromiumMapSub_1_0 PPB_OpenGLES2ChromiumMapSub_1_0_instance;
extern struct PPB_OpenGLES2DrawBuffers_Dev_1_0 PPB_OpenGLES2DrawBuffers_Dev_1_0_instance;
extern struct PPB_OpenGLES2FramebufferBlit_1_0 PPB_OpenGLES2FramebufferBlit_1_0_instance;
extern struct PPB_OpenGLES2FramebufferMultisample_1_0 PPB_OpenGLES2FramebufferMultisample_1_0_instance;
extern struct PPB_OpenGLES2InstancedArrays_1_0 PPB_OpenGLES2InstancedArrays_1_0_instance;
extern struct PPB_OpenGLES2Query_1_0 PPB_OpenGLES2Query_1_0_instance;
extern struct PPB_OpenGLES2VertexArrayObject_1_0 PPB_OpenGLES2VertexArrayObject_1_0_instance;
extern struct PPB_TCPSocket_Private_0_5 PPB_TCPSocket_Private_0_5_instance;
extern struct PPB_TextInput_Dev_0_2 PPB_TextInput_Dev_0_2_instance;
extern struct PPB_UDPSocket_Private_0_4 PPB_UDPSocket_Private_0_4_instance;
extern struct PPB_URLLoader_1_0 PPB_URLLoader_1_0_instance;
extern struct PPB_URLLoaderTrusted_0_3 PPB_URLLoaderTrusted_0_3_instance;
extern struct PPB_URLRequestInfo_1_0 PPB_URLRequestInfo_1_0_instance;
extern struct PPB_URLResponseInfo_1_0 PPB_URLResponseInfo_1_0_instance;
extern struct PPB_URLUtil_Dev_0_7 PPB_URLUtil_Dev_0_7_instance;
extern struct PPB_Var_1_2 PPB_Var_1_2_instance;
extern struct PPB_VideoCapture_Dev_0_3 PPB_VideoCapture_Dev_0_3_instance;
extern struct PPB_View_1_2 PPB_View_1_2_instance;
extern struct PPB_Printing_Dev_0_7 PPB_Printing_Dev_0_7_instance;
extern struct PPB_Var_Deprecated PPB_Var_Deprecated_instance;
extern struct PPB_BrokerTrusted_0_3 PPB_BrokerTrusted_0_3_instance;
extern struct PPB_NetworkMonitor_1_0 PPB_NetworkMonitor_1_0_instance;
extern struct PPB_MessageLoop_1_0 PPB_MessageLoop_1_0_instance;
extern struct PPB_Flash_12_6 PPB_Flash_12_6_instance;
extern struct PPB_Instance_Private_0_1 PPB_Instance_Private_0_1_instance;
extern struct PPB_VarArray_1_0 PPB_VarArray_1_0_instance;
extern struct PPB_VarArrayBuffer_1_0 PPB_VarArrayBuffer_1_0_instance;
extern struct PPB_VarDictionary_1_0 PPB_VarDictionary_1_0_instance;