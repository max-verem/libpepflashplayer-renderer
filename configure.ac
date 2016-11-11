AC_INIT(configure.ac)
AM_INIT_AUTOMAKE(libpepflashplayer-renderer, 0.1)
AM_CONFIG_HEADER(config.h)
AM_MAINTAINER_MODE
AC_PROG_RANLIB

AC_ISC_POSIX
AC_PROG_CC
AM_PROG_CC_STDC
AM_PROG_CC_C_O
AC_HEADER_STDC
AC_PROG_CXX

AC_CHECK_FUNCS(localtime_r)

LIBDEP=""
AC_CHECK_LIB(pthread, pthread_create, [LIBDEP+=-lpthread])
AC_SUBST(LIBDEP)

AM_CONDITIONAL([WIN32], [test x$win32 = xtrue])

PKG_CHECK_MODULES(CURL, [libcurl])
AC_SUBST(CURL_CFLAGS)
AC_SUBST(CURL_LIBS)

PKG_CHECK_MODULES(URIPARSER, [liburiparser])
AC_SUBST(URIPARSER_CFLAGS)
AC_SUBST(URIPARSER_LIBS)

AC_ARG_WITH([pepper],
        [AS_HELP_STRING([--with-pepper=path], [path to pepper_xx sdk])],
        [PEPPER_PATH=$withval],
        [PEPPER_PATH=/usr/local/nacl_sdk/pepper_49])
AC_SUBST(PEPPER_PATH)

PPAPI_CFLAGS="$PEPPER_PATH/include"
AC_SUBST(PPAPI_CFLAGS)

dnl PKG_CHECK_MODULES(XMLRPC, [xmlrpc xmlrpc_client xmlrpc_util])
dnl AC_SUBST(XMLRPC_CFLAGS)
dnl AC_SUBST(XMLRPC_LIBS)
AC_CHECK_PROG(have_xmlrpc_c_config, xmlrpc-c-config, yes, no)
if test $have_xmlrpc_c_config = no; then
    MUST_BUILD_XMLRPC_C=no
    AC_MSG_ERROR("Cannot find xmlrpc-c")
else
    XMLRPC_CFLAGS="`xmlrpc-c-config client abyss-server server-util --cflags`"
    XMLRPC_LIBS="`xmlrpc-c-config client abyss-server server-util --libs`"
    AC_SUBST(XMLRPC_CFLAGS)
    AC_SUBST(XMLRPC_LIBS)
fi

AM_PATH_XML2(2.4.0)

dnl Use -Wall if we have gcc.
changequote(,)dnl
if test "x$GCC" = "xyes"; then
  case " $CFLAGS " in
  *[\ \	]-Wall[\ \	]*) ;;
  *) CFLAGS="$CFLAGS -Wall" ;;
  esac
fi
changequote([,])dnl

AC_OUTPUT([
Makefile
src/Makefile
src/box1/Makefile
src/plyconv/Makefile
src/ui/Makefile
src/thecoreuicommon/library/Makefile
src/thecoreuicommon/subclip/Makefile
src/thecoreuicommon/srv/Makefile
src/thecoreuicommon/ui_lcd_images/Makefile
src/thecoreuicommon/ui_vtr/Makefile
src/thecoreuicommon/settings/Makefile
src/chyro/Makefile
src/landscape/Makefile
src/abcvs/Makefile
src/beinsports/Makefile
src/libfeeder/Makefile
src/libcleaner/Makefile
src/stl/Makefile
src/rki/Makefile
src/ticker_crawl/Makefile
])
