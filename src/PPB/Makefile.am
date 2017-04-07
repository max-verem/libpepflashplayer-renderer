noinst_LIBRARIES = libPPB.a

libPPB_a_SOURCES = \
    PPB_VarDictionary.c \
    PPB_VarArrayBuffer.c \
    PPB_VarArray.c \
    PPB_Audio.c \
    PPB_AudioConfig.c \
    PPB_AudioInput_Dev.c \
    PPB_BrokerTrusted.c \
    PPB_BrowserFont_Trusted.c \
    PPB_Buffer_Dev.c \
    PPB_CharSet_Dev.c \
    PPB_Core.c \
    PPB_Crypto_Dev.c \
    PPB_CursorControl_Dev.c \
    PPB_FileChooser_Dev.c \
    PPB_FileChooserTrusted.c \
    PPB_FileRef.c \
    PPB_Flash.c \
    PPB_Flash_Clipboard.c \
    PPB_Flash_File_FileRef.c \
    PPB_Flash_FontFile.c \
    PPB_FlashFullscreen.c \
    PPB_Flash_Menu.c \
    PPB_Graphics2D.c \
    PPB_Graphics3D.c \
    PPB_ImageData.c \
    PPB_IMEInputEvent_Dev.c \
    PPB_InputEvent.c \
    PPB_Instance.c \
    PPB_Instance_Private.c \
    PPB_Memory_Dev.c \
    PPB_MessageLoop.c \
    PPB_NetAddress_Private.c \
    PPB_NetworkMonitor.c \
    PPB_OpenGLES2.c \
    PPB_OpenGLES2ChromiumEnableFeature.c \
    PPB_OpenGLES2ChromiumMapSub.c \
    PPB_OpenGLES2DrawBuffers_Dev.c \
    PPB_OpenGLES2FramebufferBlit.c \
    PPB_OpenGLES2FramebufferMultisample.c \
    PPB_OpenGLES2InstancedArrays.c \
    PPB_OpenGLES2Query.c \
    PPB_OpenGLES2VertexArrayObject.c \
    PPB_Printing_Dev.c \
    PPB_TCPSocket_Private.c \
    PPB_TextInput_Dev.c \
    PPB_UDPSocket_Private.c \
    PPB_URLLoader.c \
    PPB_URLLoaderTrusted.c \
    PPB_URLRequestInfo.c \
    PPB_URLResponseInfo.c \
    PPB_URLUtil_Dev.c \
    PPB_Var.c \
    PPB_Var_Deprecated.c \
    PPB_VideoCapture_Dev.c \
    PPB_View.c

AM_CPPFLAGS = \
    @STD_CFLAGS@ \
    @CUDA_CFLAGS@ \
    @PPAPI_CFLAGS@ \
    -I$(top_srcdir)/src/pepper_obsolete \
    -I$(top_srcdir)/src/cuda_samples/common/inc \
    @URIPARSER_CFLAGS@ \
    @CURL_CFLAGS@ \
    @PANGOFT2_CFLAGS@ \
    -I$(top_srcdir)/src/common
