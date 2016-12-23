package com.broadcastsolutionsdesign.LiveHistory
{
    import flash.net.URLRequest;
    import flash.net.URLRequestMethod;
    import flash.net.URLLoader;
    import flash.net.URLVariables;
    import flash.net.URLLoaderDataFormat;
    import flash.events.*;

    import flash.utils.ByteArray;

    import com.broadcastsolutionsdesign.LiveHistory.Utils;
    import com.broadcastsolutionsdesign.LiveHistory.URLLoader2;

    public class FLVChunk extends Object
    {
        private var url:String = null;


        public var rt:Number;
        public var v_d:Number;
        public var id:String;
        private var v_s_t_probed:int;
        public var v_s_t:int;

        private var flv:ByteArray = null;
        private var loader:URLLoader2 = null;

        public var lru:Date = null;

        public function dispose():void
        {
            if(!flv)
                return;

            flv.clear();
            flv = null;
        };

        public function is_loading():Boolean
        {
            return (loader != null);
        };

        public function is_loaded():Boolean
        {
            return (flv != null);
        };

        public function FLVChunk(src:Object)
        {
            this.rt = Number(src["rt"]);
            this.v_d = Number(src["v_d"]);
            this.id = src["id"];
            this.v_s_t = src["v_s_t"];
            this.lru = new Date();
        };

        public function load(url:String, cb_complete:Function):void
        {
            /* create loader */
            loader = new URLLoader2
            (
                url, URLLoaderDataFormat.BINARY,
                function(e:Event):void
                {
                    Utils.trace_timed("FLVChunk.load: ul.load(" + url + ") COMPLETE " + e.target.data.length + " bytes");
                    flv = e.target.data;
                    loader.dispose();
                    loader = null;

                    // save first video timestamp
                    Utils.trace_timed("FLVChunk.load: id=" + id + ", v_s_t=" + v_s_t);
                    FLVUtils.tag_iter(flv, FLVUtils.get_header_size(flv), function(head:int, idx:int):Boolean
                    {
                        var t:int = FLVUtils.tag_type(flv, head);
                        var p:int = FLVUtils.tag_packet_type(flv, head);
                        var ts:int = FLVUtils.tag_timestamp(flv, head);

//                      Utils.trace_timed("FLVChunk.get_init_data: head=" + head + ", s=" + s + " t=" + t + ", p=" + p);

                        if(FLVUtils.TAG_TYPE_VIDEO == t && FLVUtils.TAG_PACKET_AVC_NALU == p)
                        {
                            Utils.trace_timed("FLVChunk.load: first video ts=" + ts);
                            v_s_t_probed = ts;
                            return false;
                        };

                        return true;
                    });

                    cb_complete();
                },
                function(e:IOErrorEvent):void
                {
                    Utils.trace_timed("FLVChunk.load: ul.load(" + url + ") IOErrorEvent: " + e.text);
                    flv = new ByteArray();
                    loader.dispose();
                    loader = null;
                },
                function(e:SecurityErrorEvent):void
                {
                    Utils.trace_timed("FLVChunk.load: ul.load(" + url + ") SecurityErrorEvent: " + e.toString());
                    flv = new ByteArray();
                    loader.dispose();
                    loader = null;
                },
                function(e:Error):void
                {
                    Utils.trace_timed("FLVChunk.load: ul.load(" + url + ") Error: " + e);
                    flv = new ByteArray();
                    loader.dispose();
                    loader = null;
                }
            );
            Utils.trace_timed("FLVChunk.load: ul.load(" + url + ") started");
        };

        public function get_init_data():ByteArray
        {
            var dst:ByteArray = new ByteArray();
            var h:int = FLVUtils.get_header_size(flv);

            dst.writeBytes(flv, 0, h);
            Utils.trace_timed("FLVChunk.get_init_data: head dst.lenght=" + dst.length);

            FLVUtils.tag_iter(flv, h, function(head:int, idx:int):Boolean
            {
                var t:int = FLVUtils.tag_type(flv, head);
                var s:int = FLVUtils.tag_size(flv, head);
                var p:int = FLVUtils.tag_packet_type(flv, head);

//                Utils.trace_timed("FLVChunk.get_init_data: head=" + head + ", s=" + s + " t=" + t + ", p=" + p);

                if(FLVUtils.TAG_TYPE_SCRIPT == t ||
                    (FLVUtils.TAG_TYPE_VIDEO == t && FLVUtils.TAG_PACKET_AVC_SEQ == p) ||
                    (FLVUtils.TAG_TYPE_AUDIO == t && FLVUtils.TAG_PACKET_AAC_SEQ == p))
                {
                    Utils.trace_timed("FLVChunk.get_init_data: head=" + head + ", s=" + s + " t=" + t + ", p=" + p);
//                    Utils.trace_timed("FLVChunk.get_init_data: PUSHED");
                    dst.writeBytes(flv, head, s);
                };

                return true;
            });

            lru = new Date();

            return dst;
        };

        public function correct_offset(offset:Number):Number
        {
            var head_prev:int = -1;
            var n:Number = 0;
            var h:int = FLVUtils.get_header_size(flv);

            FLVUtils.tag_iter(flv, h, function(head:int, idx:int):Boolean
            {
                var t:int = FLVUtils.tag_type(flv, head);
                var p:int = FLVUtils.tag_packet_type(flv, head);
                var ts:int = FLVUtils.tag_timestamp(flv, head);

                if(FLVUtils.TAG_TYPE_VIDEO != t || FLVUtils.TAG_PACKET_AVC_NALU != p)
                    return true;

                if((ts - v_s_t_probed) < offset)
                {
                    head_prev = head;
                    return true;
                };

                if(head_prev < 0)
                    head_prev = head;

                n = FLVUtils.tag_timestamp(flv, head_prev) - v_s_t_probed;

                return false;
            });

            return n;
        };

        public function get_cont_tags(first_video_tag_ts:Number):ByteArray
        {
            var h:int = FLVUtils.get_header_size(flv);
            var dst:ByteArray = new ByteArray();

            FLVUtils.tag_iter(flv, h, function(head:int, idx:int):Boolean
            {
                var t:int = FLVUtils.tag_type(flv, head);
                var p:int = FLVUtils.tag_packet_type(flv, head);
                var ts:int = FLVUtils.tag_timestamp(flv, head);

                if((FLVUtils.TAG_TYPE_VIDEO == t && FLVUtils.TAG_PACKET_AVC_NALU == p) ||
                    (FLVUtils.TAG_TYPE_AUDIO == t && FLVUtils.TAG_PACKET_AAC_RAW == p))
                {
                    dst.writeBytes(flv, head, flv.length - head);

                    Utils.trace_timed("FLVChunk.get_init_tags: first_video_tag_ts=" + first_video_tag_ts + ", v_s_t_probed=" + v_s_t_probed);

                    FLVUtils.tag_timestamp_update(dst, first_video_tag_ts);

                    return false;
                };

                return true;
            });

            lru = new Date();

            return dst;
        };

        public function get_init_tags(offset:Number):ByteArray
        {
            var h:int = FLVUtils.get_header_size(flv);
            var dst:ByteArray = new ByteArray();

            FLVUtils.tag_iter(flv, h, function(head:int, idx:int):Boolean
            {
                var t:int = FLVUtils.tag_type(flv, head);
                var p:int = FLVUtils.tag_packet_type(flv, head);
                var ts:int = FLVUtils.tag_timestamp(flv, head);

                if(ts >= (v_s_t_probed + offset))
                {
                    dst.writeBytes(flv, head, flv.length - head);
                    FLVUtils.tag_timestamp_update(dst, 0);
                    return false;
                };

                return true;
            });

            lru = new Date();

            return dst;
        };
    }
}