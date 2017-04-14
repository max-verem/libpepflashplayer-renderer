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

    public class DebugParams extends MovieClip
    {
        public function DebugParams()
        {
            super();

            log("DebugParams::DebugParams [here]");

            // dump
            for(var o: String in loaderInfo.parameters)
            {
                log("DebugParams::DebugParams [" + o + "]=[" + loaderInfo.parameters[o] + "]");
            };
        }

        private function log(l:String):String
        {
            Utils.trace_timed(l);
            return l;
        }
    }
}
