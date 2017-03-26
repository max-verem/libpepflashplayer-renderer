dnl
dnl autoreconf --force --install ; ./configure ; make
dnl

AC_INIT([libpepflashplayer-renderer], [0.1])
AM_INIT_AUTOMAKE([foreign])
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

LIBDEP="-lOpenGL -lEGL"
AC_SUBST(LIBDEP)

AC_CHECK_LIB([pthread], [pthread_create])
AC_CHECK_LIB([dl], [dlopen])

AM_CONDITIONAL([WIN32], [test x$win32 = xtrue])

PKG_CHECK_MODULES(CURL, [libcurl])
AC_SUBST(CURL_CFLAGS)
AC_SUBST(CURL_LIBS)

PKG_CHECK_MODULES(PANGOFT2, [pangoft2])
AC_SUBST(PANGOFT2_CFLAGS)
AC_SUBST(PANGOFT2_LIBS)

PKG_CHECK_MODULES(URIPARSER, [liburiparser])
AC_SUBST(URIPARSER_CFLAGS)
AC_SUBST(URIPARSER_LIBS)

AC_ARG_WITH([pepper],
        [AS_HELP_STRING([--with-pepper=path], [path to pepper_xx sdk])],
        [PEPPER_PATH=$withval],
        [PEPPER_PATH=/usr/local/nacl_sdk/pepper_49])
AC_SUBST(PEPPER_PATH)

PPAPI_CFLAGS="-I$PEPPER_PATH/include"
AC_SUBST(PPAPI_CFLAGS)

AC_ARG_WITH([cuda],
        [AS_HELP_STRING([--with-cuda=path], [path to cuda sdk])],
        [CUDA_PATH=$withval],
        [CUDA_PATH=/usr/local/cuda])
AC_SUBST(CUDA_PATH)

CUDA_CFLAGS="-I$CUDA_PATH/include"
AC_SUBST(CUDA_CFLAGS)

CUDA_LIBS="-L$CUDA_PATH/lib64 -lcuda -lcudart"
AC_SUBST(CUDA_LIBS)

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
src/tracer/Makefile
src/PPB/Makefile
])
