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
    import com.broadcastsolutionsdesign.libpepflashplayer_renderer.CtlLayer;

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
        private var layers_count:int = 16;
        private var layers:Array;

        private function socket_connect(delay:int):void
        {
            log("socket_connect: starting");
            var t:Timer = new Timer(delay, 2);
            t.addEventListener(TimerEvent.TIMER_COMPLETE/*TIMER*/, function(e:TimerEvent):void
            {
                log("socket_connect: connecting");
                if(socket.connected)
                    socket.close();
                socket.connect(connect_host, parseInt(connect_port));
            });
            t.start();
        };

        public function CtlProxy()
        {
            var i:int;

            log("CtlProxy: built on [" + buildDate + "] from git head [" + buildHead + "]");
            log("CtlProxy: Flash version=" + Capabilities.version + " isDebugger=" + Capabilities.isDebugger);

            super();

            Security.allowDomain(connect_host);
            Security.allowInsecureDomain(connect_host);
            Security.loadPolicyFile("http://" + connect_host + "/socket.xml");

            processParameters();

            /* init layers */
            layers = new Array(layers_count);
            for(i = 0; i < 16; i++)
            {
                var mov:CtlLayer = new CtlLayer();

                layers[i] = mov;

                addChild(mov);
            };
//            addEventListener(Event.ENTER_FRAME, function(e:Event):void
//            {
//                log("CtlProxy: ENTER_FRAME");
//            });

            stage.align = StageAlign.TOP_LEFT;
            stage.scaleMode = StageScaleMode.NO_SCALE;
            stage.addEventListener(Event.RESIZE, onStageResize);

            socket.addEventListener(Event.CONNECT, onSocketConnect);
            socket.addEventListener(Event.CLOSE, onSocketClose);
            socket.addEventListener(IOErrorEvent.IO_ERROR, onSocketError);
            socket.addEventListener(ProgressEvent.SOCKET_DATA, onSocketResponse);
            socket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, onSocketSecError);
            socket.timeout = parseInt(connect_timeout);

            socket_connect(0);
        }

        private function onSocketConnect(e:Event):void
        {
            log("onSocketConnect: here");
        }
        private function onSocketClose(e:Event):void
        {
            log("onSocketClose: here");
//            socket_connect(5000);
        }
        private function onSocketError(e:IOErrorEvent):void
        {
            log("onSocketError: " + e);
            socket_connect(1000);
        }
        private function onSocketSecError(e:SecurityErrorEvent):void
        {
            log("onSocketSecError: " + e);
//            socket_connect(1000);
        }
        private function onSocketResponse(e:ProgressEvent):void
        {
            var socket:Socket = e.target as Socket;

            log("onSocketResponse: enter");

            while(socket.bytesAvailable)
            {
                var t:int;

                log("onSocketResponse: socket.bytesAvailable=" + socket.bytesAvailable);

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

                    log("onSocketResponse: cmd=[" + cmd + "]");
                    cmd = exec_cmd(cmd);

                    socket.writeUTFBytes(cmd + ":");
                    socket.flush();
                }
            }
        }

        private function exec_cmd(body:String):String
        {
            var t:int;
            var r:String, cmd:String;

            t = body.indexOf(" ");

            if(t < 0)
                return "notparsable(exec_cmd)";

            cmd = body.substr(0, t);

            if(cmd == "LayerLoad")
                r = exec_LayerLoad(body.substr(t + 1));
            else if(cmd == "Cut")
                r = exec_Cut(body.substr(t + 1));
            else
                r = "notimplemented(exec_cmd)";

//            stage.invalidate();

            return r;
        }

        private function exec_Cut(body:String):String
        {
            var t:int, idx:int, cut:int;
            var movie:String;

            t = body.indexOf(" ");

            if(t < 0)
                return "notparsable(exec_Cut)";

            idx = parseInt(body.substr(0, t));
            cut = parseInt(body.substr(t + 1));

            log("exec_Cut: idx=[" + idx + "], cut=[" + cut + "]");

            layers[idx].Cut(cut);

            return "OK(exec_Cut)";
        }

        private function exec_LayerLoad(body:String):String
        {
            var t:int, idx:int;
            var movie:String;

            t = body.indexOf(" ");

            if(t < 0)
                return "notparsable(exec_LayerLoad)";

            idx = parseInt(body.substr(0, t));
            movie = body.substr(t + 1);

            log("exec_LayerLoad: idx=[" + idx + "], movie=[" + movie + "]");

            layers[idx].Load(movie);

            return "OK(exec_LayerLoad)";
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
