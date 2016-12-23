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
    import flash.net.URLRequest;
    import flash.net.URLLoader;
    import flash.net.URLLoaderDataFormat;
    import flash.net.URLRequestMethod;

    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.Utils;
    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.URLLoader2;

    public class as3_test3 extends MovieClip
    {
        [Embed(source="../../../res/m1-hb.ttf",
            fontName = "myFont",
            mimeType = "application/x-font",
            advancedAntiAliasing="true",
            embedAsCFF="false")]
        private var myEmbeddedFont:Class;

        private static const buildDate:String = NAMES::BuildDate;
        private static const buildHead:String = NAMES::BuildHead;

        private var param1:String = "";

        public function as3_test3()
        {
            stage.align = StageAlign.TOP_LEFT;
            stage.scaleMode = StageScaleMode.NO_SCALE;
            stage.addEventListener(Event.RESIZE, onStageResize);
            super();
            processParameters();
            init();
            log("constructor finished");
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
            log("processParameters: entering");

            if(loaderInfo.parameters.param1)
            {
                param1 = loaderInfo.parameters.param1;

                log("processParameters: param1=[" + param1 + "]");
            };

        }

        private function init():void
        {
            log("Init: built on [" + buildDate + "] from git head [" + buildHead + "]");
            log("Init Done: Flash version=" + Capabilities.version + " isDebugger=" + Capabilities.isDebugger);
            log("Load started");

            var textField: TextField = new TextField();
            textField.defaultTextFormat = new TextFormat("myFont", 80);
            textField.embedFonts = true;
            textField.text = param1;
            textField.textColor= 0x000000;
            textField.selectable= false;
            textField.blendMode = BlendMode.LAYER;
            textField.autoSize = TextFieldAutoSize.LEFT;

            log("init: textField.width=[" + textField.width + "], textField.heigh=[" + textField.height + "]");

            log("Load started1");

            addChild(textField);

            log("Load started2");
            log("Load started3");
            log("Load started4");

            app_call
            (
                "http://dev-5.internal.m1stereo.tv/demo.json",
                function(args:Object):void
                {
                    args.test1 = "test";
                    args.string = param1;
                },
                function(resp:Object):void
                {
                    log("response demo=[" + resp['demo'] + "]");
                    log("response got2");
                }
            );

            log("Load started5");
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

        private function app_call(app_url:String, argset:Function, callback:Function):void
        {
            /* init object to send */
            var json:Object;
            json = new Object();
            argset(json); /* set other parameter */

            /* init request */
            var req:URLRequest = new URLRequest(app_url);
            req.method = URLRequestMethod.POST;
            req.data = JSON.stringify(json);
            req.contentType = "text/x-json";

            /* init loader */
            var ul:URLLoader = new URLLoader();
            ul.dataFormat = URLLoaderDataFormat.TEXT;
            ul.addEventListener(Event.COMPLETE, function(e:Event):void
            {
                var er:Error = null;
                var j:Object = null;

                log("app_call: Event.COMPLETE=" + e.target.data);

                try
                {
                    j = JSON.parse(e.target.data);
                    log("app_call: e.target.data=" + e.target.data);
                }
                catch(e:Error)
                {
                    er = e;
                };

                if(j == null)
                    log("app_call: PARSE ERROR of JSON response:" + er);
                else
                {
                    callback(j);
                };

                log("app_call: DONE");
            });
            ul.addEventListener(IOErrorEvent.IO_ERROR, function(e:IOErrorEvent):void
            {
                log("app_call: IO ERROR:" + e.text);
                log("app_call: IO ERROR:" + e.text);
            });

            log("app_call: calling:" + JSON.stringify(json));
            try
            {
                ul.load(req);
            }
            catch(e:Error)
            {
                log("app_call: CALL ERROR:" + e);
                log("app_call: CALL ERROR:" + e);
            };
        }


    }
}
