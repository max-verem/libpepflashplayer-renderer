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
    import flash.events.Event;
    import flash.display.MovieClip;
    import flash.display.DisplayObject;

    import flash.events.ProgressEvent;
    import flash.net.Socket;

    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.Utils;
    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.URLLoader2;

    [SWF(frameRate=50)]

    public class CtlLayer extends MovieClip
    {
        private var loader:Loader = null;
        private var movie:DisplayObject = null;

        public function CtlLayer()
        {
            super();
            this.alpha = 0.0;
        }

        public function Cut(v:int):void
        {
            var mc:MovieClip;

            if(v)
            {
                if(movie && (movie is MovieClip)) { mc = movie as MovieClip; mc.gotoAndPlay(0); }

                this.alpha = 1.0;
            }
            else
            {
                this.alpha = 0.0;

                if(movie && (movie is MovieClip)) { mc = movie as MovieClip; mc.stop(); }
            };
        };

        public function Load(name:String):void
        {
            var url:URLRequest = new URLRequest(name);

            if(loader)
            {
                try
                {
                    loader.close();
                }
                catch (myError:Error)
                {
                    log("loader.close(): " + myError);
                };

                loader = null;
            };
            loader = new Loader();
            loader.uncaughtErrorEvents.addEventListener(UncaughtErrorEvent.UNCAUGHT_ERROR, function(event:UncaughtErrorEvent):void
            {
                log("CtlLayer:Load: loader error");

                loader = null;
            });
            loader.contentLoaderInfo.addEventListener(Event.COMPLETE, function(event:Event):void
            {
                var li:LoaderInfo = event.currentTarget as LoaderInfo;
                var mc:DisplayObject = li.content as DisplayObject;

                log("CtlLayer:Load: contentType=[" + li.contentType + "]");

                if("application/x-shockwave-flash" == li.contentType)
                {
                    log("CtlLayer:Load: swfVersion=[" + li.swfVersion + "]");
                };

                if(movie)
                {
                    removeChild(movie);
                    movie = null;
                };

                // try to add as children
                try
                {
                    log("CtlLayer:Load: QualifiedClassName=[" + flash.utils.getQualifiedClassName(mc) + "]");

                    if(mc is AVM1Movie)
                    {
                        log("CtlLayer:Load: AVM1Movie");
                        mc = li as DisplayObject;
                    };

                    if(mc == null)
                        log("CtlLayer:Load: (mc == null)");
                    else
                        addChild(mc);

                    movie = mc;
                }
                catch (myError:Error)
                {
                    log("addChild(): " + myError);
                };

                log("CtlLayer:Load: child added");

                loader = null;
            });
            loader.load(url);
        }

        private function log(l:String):String
        {
            Utils.trace_timed(l);
            return l;
        }
    }
}
