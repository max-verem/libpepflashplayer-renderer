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
    import flash.net.LocalConnection;
    import flash.net.NetConnection;         /* http://help.adobe.com/en_US/AS3LCR/Flash_10.0/flash/net/NetConnection.html */
    import flash.net.NetStream;             /* http://help.adobe.com/en_US/AS3LCR/Flash_10.0/flash/net/NetStream.html */
    import flash.media.Video;               /* http://help.adobe.com/en_US/AS3LCR/Flash_10.0/flash/media/Video.html */
    import flash.utils.ByteArray;
    import flash.net.NetStreamAppendBytesAction;
    import flash.system.Capabilities;

    import com.broadcastsolutionsdesign.LiveHistory.Utils;
    import com.broadcastsolutionsdesign.LiveHistory.FLVChunk;

    public class player extends MovieClip
    {
        private var js_prefix:String = "";
        private var f_debug:int = 0;
        private var max_loaders:int = 3;
        private var preload_dur:Number = 30;
        private var chunks_count_max:int = 10;

        private static const buildDate:String = NAMES::BuildDate;
        private static const buildHead:String = NAMES::BuildHead;

        private var vid:Video;
        private var nc:NetConnection;
        private var ns:NetStream;
        private var ns_client:Object;
        private var ns_init_done:int = 0;

        private var notify_timer:Timer;

        private var txtLog: TextField = new TextField();

        private var seek_to:Number = -1.0;
        private var seek_pos:Number = -1.0;
        private var seek_length:Number = -1.0;
        private var seek_rt:Number = -1.0;
        private var chunks:Array = new Array();

        public function player()
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
            if(loaderInfo.parameters.js_prefix)
                js_prefix = loaderInfo.parameters.js_prefix;
            if(loaderInfo.parameters.f_debug)
                f_debug = loaderInfo.parameters.f_debug;
            if(loaderInfo.parameters.max_loaders)
                max_loaders = loaderInfo.parameters. max_loaders;
            if(loaderInfo.parameters.preload_dur)
                preload_dur = loaderInfo.parameters. preload_dur;
            if(loaderInfo.parameters.chunks_count_max)
                chunks_count_max = loaderInfo.parameters. chunks_count_max;
        }

        private function chunks_cleanup():void
        {
            var i:int, j:int;

            if(chunks.length < chunks_count_max)
                return;

            log("chunks_cleanup: chunks.length=" + chunks.length + " chunks_count_max=" + chunks_count_max);

            for(i = 0, j = 1; j < chunks.length; j++)
                if(chunks[i].lru.time > chunks[j].lru.time)
                    i = j;

            var now:Date = new Date();
            var chunk:FLVChunk = chunks[i];

            /* check if it is loading */
            if(chunk.is_loading())
                return;

            log("chunks_cleanup: i=" + i + ", age=" + ((now.time - chunk.lru.time) / 86400)+ " seconds");

            /* delete from array */
            chunks.splice(i, 1);

            /* dispose a buffers */
            chunk.dispose();
            chunk = null;
        };

        private function init():void
        {
            init_log();
            log("Init: built on [" + buildDate + "] from git head [" + buildHead + "]");
            log("Init Done: Flash version=" + Capabilities.version + " isDebugger=" + Capabilities.isDebugger);
            log("Load started");

            init_video();

            /* init external interface callbacks */
            ExternalInterface.addCallback("toggle_play", js_toggle_play);
            ExternalInterface.addCallback("pause", js_pause);
            ExternalInterface.addCallback("resume", js_resume);
            ExternalInterface.addCallback("seek_rel", js_seek_rel);
            ExternalInterface.addCallback("seek_to", js_seek_to);
            ExternalInterface.addCallback("get_current_position", js_get_current_position);

            notify_timer = new Timer(100);
            notify_timer.addEventListener(TimerEvent.TIMER, function(e:TimerEvent):void
            {
                /* notify parent html about positions */
                try
                {
                    ExternalInterface.call(js_prefix + "notify_time",
                        js_get_current_position(),      // current position in astronimic
                        seek_length,                    // how much seconds sent to
                        ns.time, ns.bufferLength,       // NetStream params,
                        seek_to, seek_pos, seek_length, seek_rt);
                }
                catch(e:Error)
                {
                    Utils.trace_timed(e.toString());
                };

                /* wakeup processing */
                wakeup();

                /* chunks cleanup */
//                chunks_cleanup();
            });
            notify_timer.start();

            /* notify html that init done */
            ExternalInterface.call(js_prefix + "init_done");
        }

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

        private function log(l:String):String
        {
            Utils.trace_timed(l);
            if(f_debug)
                txtLog.text = l + "\n" + txtLog.text;
            return l;
        }

        private function log2(l:String):String
        {
            return l;
        }

        private function init_video():void
        {
            ns_client = new Object();
            ns_client.onImageData = function(p:Object):void  { log("onImageData"); };
            ns_client.onMetaData = function(p:Object):void { log("onMetaData: width="
                + p.width + " framerate=" + p.framerate); };
            ns_client.onBWDone = function():Boolean { log("onBWDone"); return true; };
            ns_client.onBWCheck = function():Number { log("onBWCheck"); return 0; };
            ns_client.onCuePoint = function(p:Object):void { log("onCuePoint"); };
            ns_client.onTextData = function(p:Object):void { log("onTextData"); };
            ns_client.onPlayStatus = function(p:Object):void { log("onPlayStatus"); };

            nc = new NetConnection();
            nc.connect(null);
            ns = new NetStream(nc);
            ns.addEventListener(NetStatusEvent.NET_STATUS, netStatusHandler, false, 0, true);
            ns.addEventListener(SecurityErrorEvent.SECURITY_ERROR, netStatusHandler, false, 0, true);
            ns.addEventListener(AsyncErrorEvent.ASYNC_ERROR, netStatusHandler, false, 0, true);
            ns.client = ns_client;
            ns.play(null);
            ns.pause();

            vid = new Video(640, 360);
            vid.smoothing = true;
            addChild(vid);
            setChildIndex(vid, 0);
            vid.attachNetStream(ns);

            /* setup low latency of audio: http://www.bytearray.org/?p=3705 */
            ns.useJitterBuffer = true;
            ns.bufferTime = 0;
        };

        private function netStatusHandler(event:NetStatusEvent):void
        {
            log("netStatusHandler: " + event.info.code);

            try
            {
                switch (event.info.code) 
                {
                    case "NetConnection.Connect.Closed":
                    case "NetConnection.Connect.Failed":
                    case "NetConnection.Connect.Rejected":
                    break;

                    case "NetConnection.Connect.Success":
                    break;

                    case "NetStream.Play.Start":
                    break;

                    case "NetStream.Publish.BadName":
                    case "NetStream.Play.StreamNotFound":
                    break;

                    case "NetStream.Play.Stop":
                    break;

                    case "NetStream.Seek.Notify":
                    break;

                    case "NetStream.Buffer.Full":
//                    seek_wait_hide();
                    break;
                }
            }
            catch (error:TypeError)
            {
                // Ignore any errors.
                log("netStatusHandler: " + error);
            };
        };

        private function js_seek_to(t:Number):void
        {
            var i:int, j:int;
            var r:Object;
            var f:FLVChunk;

            log("js_seek_to: t=" + t);

            /* request chunk idx that belongs to that time */
            i = ExternalInterface.call(js_prefix + "find_chunk_idx", t);
            log("js_seek_to: idx=" + i);

            /* get chunk info */
            r = ExternalInterface.call(js_prefix + "get_chunk_desc", i);
            log("js_seek_to: r[rt]=" + r["rt"]);
            log("js_seek_to: r[id]=" + r["id"]);

            /* create new chunk object */
            f = new FLVChunk(r);

            /* clip it */
            if(t < f.rt)
                t = f.rt;

            /* register new chunk if not exist */
            chunk_lookup_rt(t, f);

            seek_to = t;
            wakeup();
        };

        private function chunk_lookup_rt(t:Number, new_chunk_obj:FLVChunk):int
        {
            var i:int, j:int;

            /* look throw cached item to check if it present */
            for(i = 0, j = -1; i < chunks.length && j == -1; i++)
                if(chunks[i].rt <= t && t < (chunks[i].rt + chunks[i].v_d))
                    j = i;
            if(j >= 0)
                return j;

            if(new_chunk_obj == null)
                return -1;

            /* avoid duplicating item to insert */
            for(i = 0, j = -1; i < chunks.length && j == -1; i++)
                if(new_chunk_obj.id == chunks[i].id)
                    j = i;
            if(j == -1)
                chunks.push(new_chunk_obj);

            return chunks.length - 1;
        };

        private function wakeup():void
        {
            var i:int, j:int;
            var url:String;

            /* was seek requested */
            if(seek_to > 0)
            {
                var chunk:FLVChunk;

                log("wake up: seek_to > 0");

                /* cached chunk info */
                chunk = chunks[chunk_lookup_rt(seek_to, null)];

                /* check id data present */
                if(chunk.is_loaded())
                {
                    var d0:Number, d1:Number;
                    var buf:ByteArray;

                    /* data present - do next job */

                    /* reset */
                    ns.seek(0);
                    ns.appendBytesAction(NetStreamAppendBytesAction.RESET_SEEK);

                    /* check if video was initialized */
                    if(0 == ns_init_done)
                    {
                        /* set flag */
                        ns_init_done = 1;

                        /* append header and first video chunk */
                        buf = chunk.get_init_data();
                        ns.appendBytes(buf);
                        buf.clear();
                        buf = null;
                    };

                    /* find/correct offset */
                    d0 = seek_to - chunk.rt;
                    d1 = chunk.correct_offset(d0);
                    log("wakeup: correction delta " + d0 + " => " + d1);

                    /* set pos */
                    seek_pos = chunk.rt + d1;
                    log("wakeup: seek_to=" + seek_to + " => seek_pos=" + seek_pos);
                    seek_to = -1;

                    /* upload data */
                    buf = chunk.get_init_tags(d1);
                    ns.appendBytes(buf);
                    buf.clear();
                    buf = null;

                    /* save number of miliseconds loaded */
                    seek_length = chunk.v_d - d1;
                }
                else
                {
                    /* check if it pending loading */
                    if(chunk.is_loading())
                    {
                        /* it is already loading - ignore - wait till it be loaded */
                        return;
                    };

                    /* check if loaders count reached */
                    for(i = 0, j = 0; i < chunks.length; i++)
                        if(chunks[i].is_loading())
                            j++;
                    if(j < max_loaders)
                    {
                        /* get url */
                        url = ExternalInterface.call(js_prefix + "get_chunk_url", chunk.id);
                        chunk.load(url, function():void { /* wakeup(); */ });
                    };
                };

                chunk = null;

                return;
            };

            log("wakeup: seek_length= "  + seek_length);
            if(seek_length <= 0)
                return;

            log("wakeup: preload_dur= "  + preload_dur + " ns.bufferLength=" + ns.bufferLength);
            if(preload_dur <= ns.bufferLength)
                return;

            /* calculate future timestamp */
            var next_rt:Number = seek_pos + seek_length;
            log("wakeup: seek_pos="  + seek_pos + ", seek_length=" + seek_length);

            /* check if it present in our cache */
            log("wakeup: next_rt= "  + next_rt);
            var idx:int = chunk_lookup_rt(next_rt, null);
            log("wakeup: idx=" + idx);
            if(idx < 0)
            {
                var r:Object;

                log("wakeup: no chunk for rt=" + next_rt);

                /* request chunk idx that belongs to that time */
                i = ExternalInterface.call(js_prefix + "find_chunk_idx", next_rt);
                log("wakeup: next chunk idx=" + i);

                /* get chunk info */
                r = ExternalInterface.call(js_prefix + "get_chunk_desc", i);
                log("wakeup: next chunk r[rt]=" + r["rt"]);
                log("wakeup: next chunk r[id]=" + r["id"]);
                log("wakeup: next chunk r[v_d_ts]=" + r["v_d_ts"]);
                log("wakeup: next chunk r[v_d]=" + r["v_d"]);

                /* create new chunk object */
                chunk = new FLVChunk(r);

                /* register new chunk if not exist */
                idx = chunk_lookup_rt(next_rt, chunk);
                log("wakeup: chunk_lookup_rt=" + idx);

                /* return from processing */
                return;
            };

            /* cached chunk info */
            chunk = chunks[idx];

            /* check id data present */
            if(chunk.is_loaded())
            {
                /* loaded - append data */

                /* upload data */
                buf = chunk.get_cont_tags(seek_length);
                ns.appendBytes(buf);
                buf.clear();
                buf = null;

                /* save number of miliseconds loaded */
                seek_length += chunk.v_d;

                return;
            };

            /* check if it pending loading */
            if(chunk.is_loading())
            {
                log("wakeup: chunk.is_loading");

                /* it is already loading - ignore - wait till it be loaded */
                return;
            };

            /* check if loaders count reached */
            for(i = 0, j = 0; i < chunks.length; i++)
                if(chunks[i].is_loading())
                    j++;
            log("wakeup: j=" + j + " max_loaders=" + max_loaders);
            if(j < max_loaders)
            {
                /* get url */
                url = ExternalInterface.call(js_prefix + "get_chunk_url", chunk.id);
                chunk.load(url, function():void { wakeup(); });
            };
        };

        private function js_toggle_play():void { ns.togglePause(); };
        private function js_pause():void { ns.pause(); };
        private function js_resume():void { ns.resume(); };
        private function js_seek_rel(rel:int):void
        {
            log(Utils.trace_timed("js_seek_rel: rel=" + rel));
            js_seek_to(js_get_current_position() + rel);
        };
        private function js_get_current_position():Number
        {
            return seek_pos + ns.time * 1000;
        };

    }
}
