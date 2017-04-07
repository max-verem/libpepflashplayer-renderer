AM_LDFLAGS = -lm
AM_CFLAGS = -fms-extensions \
    @STD_CFLAGS@ \
    @CUDA_CFLAGS@ \
    @PPAPI_CFLAGS@ \
    @NDI_CFLAGS@ \
    -I$(top_srcdir)/src/cuda_samples/common/inc \
    -I$(top_srcdir)/src/pepper_obsolete \
    -I$(top_srcdir)/src/PPB \
    -I$(top_srcdir)/src/common

bin_PROGRAMS = tracer

tracer_SOURCES = \
    main.c \
    app.c \
    app_buf.c \
    app_class.c \
    app_cuda.cu \
    $(top_srcdir)/src/common/ticker.c \
    $(top_srcdir)/src/common/if.c \
    $(top_srcdir)/src/common/log.c \
    $(top_srcdir)/src/common/mod.c \
    $(top_srcdir)/src/common/res.c

.cu.o:
	$(NVCC) -o $@ -c $< --gpu-architecture=compute_50 --gpu-code=compute_50,sm_50 @CUDA_CFLAGS@ -I$(top_srcdir)/src/cuda_samples/common/inc

#$(top_srcdir)/src/PPB/libPPB.a:

SUBDIRS = ../PPB

tracer_LDADD = \
    $(top_srcdir)/src/PPB/libPPB.a \
    @CUDA_LIBS@ \
    @PANGOFT2_LIBS@ \
    @CURL_LIBS@ \
    @URIPARSER_LIBS@ \
    @NDI_LIBS@ \
    @LIBDEP@
