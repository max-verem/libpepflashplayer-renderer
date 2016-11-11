AM_CFLAGS = -fms-extensions
AM_LDFLAGS = -lm

INCLUDES = \
    @PPAPI_CFLAGS@ \
    -I$(top_srcdir)/src/pepper_obsolete \
    -I$(top_srcdir)/src \
    -I$(top_srcdir)/src/common

bin_PROGRAMS = tracer

tracer_SOURCES = \
    main.c \
    $(top_srcdir)/src/common/if.c \
    $(top_srcdir)/src/common/mod.c \
    $(top_srcdir)/src/common/res.c

#$(top_srcdir)/src/PPB/libPPB.a:

SUBDIRS = $(top_srcdir)/src/PPB

tracer_LDADD = \
    $(top_srcdir)/src/PPB/libPPB.a \
    @CURL_LIBS@ \
    @URIPARSER_LIBS@ \
    @LIBDEP@
