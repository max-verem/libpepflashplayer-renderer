package com.broadcastsolutionsdesign.LiveHistory
{
    import flash.utils.ByteArray;

    public class FLVUtils extends Object
    {
        public static function _min(A:int, B:int):int {if(A < B) return A; return B;};

        public static function r24be(buf:ByteArray, offset:int):int
        {
            var r:int = 0;

            r += buf[offset + 0];
            r <<= 8;

            r += buf[offset + 1];
            r <<= 8;

            r += buf[offset + 2];

            return r;
        };

        public static function r32be(buf:ByteArray, offset:int):int
        {
            var r:int = 0;

            r += buf[offset + 0];
            r <<= 8;

            r += buf[offset + 1];
            r <<= 8;

            r += buf[offset + 2];
            r <<= 8;

            r += buf[offset + 3];

            return r;
        };

        public static const TAG_TYPE_AUDIO:int = 8;
        public static const TAG_TYPE_VIDEO:int = 9;
        public static const TAG_TYPE_SCRIPT:int = 18;
        public static function tag_type(src:ByteArray, head:int):int
        {
            return src[head];
        };

        public static const TAG_PACKET_AAC_SEQ:int = 0;
        public static const TAG_PACKET_AAC_RAW:int = 1;
        public static const TAG_PACKET_AVC_SEQ:int = 0;
        public static const TAG_PACKET_AVC_NALU:int = 1;
        public static const TAG_PACKET_AVC_EOS:int = 2;
        public static function tag_packet_type(src:ByteArray, head:int):int
        {
            return src[head + 12];
        };

        public static function tag_data_size(src:ByteArray, head:int):int
        {
            return FLVUtils.r24be(src, head + 1);
        };

        public static function tag_size(src:ByteArray, head:int):int
        {
            return 11 /* header */ + FLVUtils.tag_data_size(src, head) + 4 /* 32-bit value of prev tag size */;
        };

        public static function get_header_size(src:ByteArray):int
        {
            return FLVUtils.r32be(src, 5) + 4;
        };

        public static function tag_timestamp(buf:ByteArray, head:int):int
        {
            return FLVUtils.r24be(buf, head + 4) + (buf[head + 7] << 24);
        };


        public static function tag_timestamp_update(buf:ByteArray, start_timestamp:int):void
        {
            var delta_timestamp:int;

            FLVUtils.tag_iter(buf, 0, function(head:int, idx:int):Boolean
            {
                var old_timestamp:int = FLVUtils.tag_timestamp(buf, head), new_timestamp:int;

                if(0 == idx)
                    delta_timestamp = start_timestamp - old_timestamp;

                new_timestamp = old_timestamp + delta_timestamp;

                buf[head + 4] = (new_timestamp >> 16) & 0xFF;
                buf[head + 5] = (new_timestamp >>  8) & 0xFF;
                buf[head + 6] = (new_timestamp >>  0) & 0xFF;
                buf[head + 7] = (new_timestamp >> 24) & 0xFF;

                return true;
            });
        };

        public static function tag_iter(buf:ByteArray, head:int, cb_tag_iter:Function):void
        {
            var idx:int = 0;
            var cont:Boolean = true;

            for(; head < buf.length && cont; idx++)
            {
                cont = cb_tag_iter(head, idx);
                head += FLVUtils.tag_size(buf, head);
            };
        };
    }
}
