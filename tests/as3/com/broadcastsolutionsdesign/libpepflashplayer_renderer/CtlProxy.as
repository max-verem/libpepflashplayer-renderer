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

    import flash.events.ProgressEvent;
    import flash.net.Socket;

    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.Utils;
    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.URLLoader2;

    [SWF(frameRate=50)]

    public class CtlProxy extends MovieClip
    {
        private static const buildDate:String = NAMES::BuildDate;
        private static const buildHead:String = NAMES::BuildHead;

        private var connect_host:String = "127.0.0.1";
        private var connect_port:String = "10000";
        private var connect_timeout:String = "10";
        private var socket_buffer:String = "";
        private var socket:Socket = new Socket();

        public function CtlProxy()
        {
            log("CtlProxy: built on [" + buildDate + "] from git head [" + buildHead + "]");
            log("CtlProxy: Flash version=" + Capabilities.version + " isDebugger=" + Capabilities.isDebugger);

            super();

            processParameters();

            stage.align = StageAlign.TOP_LEFT;
            stage.scaleMode = StageScaleMode.NO_SCALE;
            stage.addEventListener(Event.RESIZE, onStageResize);

            socket.addEventListener(Event.CONNECT, onSocketConnect);
            socket.addEventListener(Event.CLOSE, onSocketClose);
            socket.addEventListener(IOErrorEvent.IO_ERROR, onSocketError);
            socket.addEventListener(ProgressEvent.SOCKET_DATA, onSocketResponse);
            socket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, onSocketSecError);
            socket.timeout = parseInt(connect_timeout);
            socket.connect(connect_host, parseInt(connect_port));
        }

        private function onSocketConnect(e:Event):void
        {
            log("onSocketConnect: here");
        }
        private function onSocketClose(e:Event):void
        {
            log("onSocketClose: here");
        }
        private function onSocketError(e:IOErrorEvent):void
        {
            log("onSocketError: " + e);
            socket.connect(connect_host, parseInt(connect_port));
        }
        private function onSocketSecError(e:SecurityErrorEvent):void
        {
            log("onSocketError: " + e);
            socket.connect(connect_host, parseInt(connect_port));
        }
        private function onSocketResponse(e:ProgressEvent):void
        {
            var socket:Socket = e.target as Socket;

            log("SocketResponse: enter");

            while(socket.bytesAvailable)
            {
                var t:int;

                log("SocketResponse: socket.bytesAvailable=" + socket.bytesAvailable);

                /* append buffer */
                socket_buffer += (socket.readUTFBytes(socket.bytesAvailable)).toString();

                /* check terminator */
                while(1)
                {
                    var cmd:String;

                    t = socket_buffer.indexOf(":");

                    if(t < 0)
                        break;

                    cmd = socket_buffer.substr(0, t);
                    socket_buffer = socket_buffer.substr(t + 1);

                    log("SocketResponse: cmd=[" + cmd + "]");

                    socket.writeUTFBytes("{" + cmd + "}:");
                    socket.flush();
                }
            }
        }


        private function onStageResize(event: Event = null): void
        {
            try
            {
            }
            catch (e: ArgumentError)
            {
            }
        }




        private function processParameters(): void
        {
            log("processParameters: entering");

            if(loaderInfo.parameters.connect_host)
            {
                connect_host = loaderInfo.parameters.connect_host;
                log("processParameters: connect_host=[" + connect_host + "]");
            };

            if(loaderInfo.parameters.connect_port)
            {
                connect_port = loaderInfo.parameters.connect_port;
                log("processParameters: connect_port=[" + connect_port + "]");
            };
        }

        private function log(l:String):String
        {
            Utils.trace_timed(l);
            return l;
        }
    }
}
