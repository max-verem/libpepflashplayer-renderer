#!/bin/bash

#    -compiler.include-libraries $FLEX_HOME/lib/custom/as3crypto.swc \
#    -compiler.include-libraries $FLEX_HOME/lib/custom/as3corelib.swc \
#    -compiler.include-libraries $FLEX_HOME/lib/custom/as3httpclientlib-1_0_6.swc \

#    -compiler.include-libraries $FLEX_HOME/frameworks/libs/air/airglobal.swc \
#    -compiler.include-libraries $FLEX_HOME/frameworks/libs/air/aircore.swc \
#    -compiler.include-libraries $FLEX_HOME/frameworks/libs/air/airframework.swc \

BUILD_DATE=`date`
BUILD_HEAD=`git describe --always`

mxmlc \
    -define+=NAMES::BuildDate,"\"$BUILD_DATE\"" \
    -define+=NAMES::BuildHead,"\"$BUILD_HEAD\"" \
    -omit-trace-statements=false -use-network=true \
    -compiler.source-path=. \
    com/broadcastsolutionsdesign/libpepflashplayer_renderer/CtlProxy.as \
    -output ../src/CtlProxy.swf
