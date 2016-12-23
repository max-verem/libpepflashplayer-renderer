package com.broadcastsolutionsdesign.libpepflashplayer_renderer
{
    import flash.events.*;
    import flash.net.URLRequest;
    import flash.net.URLLoader;
    import flash.net.URLLoaderDataFormat;

    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.Utils;

    public class URLLoader2 extends Object
    {
        private var url:String = null;
        private var ul:URLLoader = null;
        private var cb_complete:Function = null;
        private var cb_io_error:Function = null;
        private var cb_sec_error:Function = null;

        public function URLLoader2
        (
            url:String,
            fmt:String,
            cb_complete:Function,
            cb_io_error:Function,
            cb_sec_error:Function,
            load_ex:Function
        )
        {
            /* save handlers */
            this.cb_complete = cb_complete;
            this.cb_io_error = cb_io_error;
            this.cb_sec_error = cb_sec_error;
            this.url = url;

            ul = new URLLoader();

            ul.dataFormat = fmt;

            ul.addEventListener(Event.COMPLETE, complete);
            ul.addEventListener(IOErrorEvent.IO_ERROR, io_error);
            ul.addEventListener(SecurityErrorEvent.SECURITY_ERROR, sec_error);

            var req:URLRequest = new URLRequest(url);

            try
            {
                ul.load(req);
            }
            catch(e:Error)
            {
                if(load_ex != null)
                    load_ex(e);
                else
                    dispose();
            };

            req = null;
        };

        private function complete(e:Event):void
        {
            if(cb_complete != null)
                cb_complete(e);
            else
                Utils.trace_timed("URLLoader2::complete: unhandled Event.COMPLETE for " + url);

            dispose();
        };

        private function io_error(e:IOErrorEvent):void
        {
            if(cb_io_error != null)
                cb_io_error(e);
            else
                Utils.trace_timed("URLLoader2::io_error: unhandled IOErrorEvent.IO_ERROR for " + url);

            dispose();
        };

        private function sec_error(e:SecurityErrorEvent):void
        {
            if(cb_sec_error != null)
                cb_sec_error(e);
            else
                Utils.trace_timed("URLLoader2::sec_error: unhandled SecurityErrorEvent.SECURITY_ERROR for " + url);

            dispose();
        };

        public function dispose():void
        {
            if(ul == null)
                return;

            ul.removeEventListener(Event.COMPLETE, complete);
            ul.removeEventListener(IOErrorEvent.IO_ERROR, io_error);
            ul.removeEventListener(SecurityErrorEvent.SECURITY_ERROR, sec_error);

            ul.close();
            ul = null;

            url = null;

            this.cb_complete = null;
            this.cb_io_error = null;
            this.cb_sec_error = null;
        };
    }
}
