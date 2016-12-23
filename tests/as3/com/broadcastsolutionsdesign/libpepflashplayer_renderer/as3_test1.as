package com.broadcastsolutionsdesign.libpepflashplayer_renderer
{
    import flash.system.*;
    import flash.external.*;
    import flash.display.*;
    import flash.geom.Point;
    import flash.ui.Keyboard;
    import flash.events.*;
    import flash.utils.*;
    import flash.text.*;

    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.Utils;

    public class as3_test1 extends MovieClip
    {
        private static const buildDate:String = NAMES::BuildDate;
        private static const buildHead:String = NAMES::BuildHead;

        private var param1:String = "";

        public function as3_test1()
        {
            stage.align = StageAlign.TOP_LEFT;
            stage.scaleMode = StageScaleMode.NO_SCALE;
            stage.addEventListener(Event.RESIZE, onStageResize);
            super();
            processParameters();
            init();
        }

        private function onStageResize(event: Event = null): void
        {
            try
            {
/*
                getChildIndex(txtLoadProgress);
                txtLoadProgress.x= stage.stageWidth / 2;
                txtLoadProgress.y= stage.stageHeight / 2;
*/
            }
            catch (e: ArgumentError)
            {
            }

            try
            {
/*
                getChildIndex(txtHelpMessage);
                txtHelpMessage.x= (stage.stageWidth - txtHelpMessage.textWidth) / 2;
                txtHelpMessage.y= (stage.stageHeight - txtHelpMessage.textHeight) / 2;
*/
            }
            catch (e: ArgumentError)
            {
            }
        }

        private function processParameters(): void
        {
            if(loaderInfo.parameters.param1)
                param1 = loaderInfo.parameters.param1;
        }

        private function init():void
        {
            log("Init: built on [" + buildDate + "] from git head [" + buildHead + "]");
            log("Init Done: Flash version=" + Capabilities.version + " isDebugger=" + Capabilities.isDebugger);
            log("Load started");
        }

/*
        private function init_log():void
        {
            var txtLogFormat: TextFormat = new TextFormat();
            txtLogFormat.font= "Arial";
            txtLogFormat.size= 12;
            txtLogFormat.bold= true;
            txtLog.textColor= 0x000000;
            txtLog.selectable= false;
            txtLog.defaultTextFormat= txtLogFormat;
            txtLog.autoSize= TextFieldAutoSize.LEFT;
            txtLog.blendMode = BlendMode.LAYER;
            if(f_debug)
                txtLog.alpha = 0.8;
            else
                txtLog.alpha = 0.0;
            addChild(txtLog);
        }
*/
        private function log(l:String):String
        {
            Utils.trace_timed(l);
            return l;
        }
    }
}
