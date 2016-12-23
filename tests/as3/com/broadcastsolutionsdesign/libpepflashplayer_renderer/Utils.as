package com.broadcastsolutionsdesign.libpepflashplayer_renderer
{
    public class Utils extends Object
    {
        public static function trace_timed(msg:String):String
        {

            var dateObj:Date = new Date();

            var year:String = String(dateObj.getFullYear());

            var mon:String = String(dateObj.getMonth() + 1);
            if(mon.length == 1)
                mon = "0" + mon;

            var day:String = String(dateObj.getDate());
            if(day.length == 1)
                day = "0" + day;

            var hours:String = String(dateObj.getHours());
            if(hours.length == 1)
                hours = "0" + hours;

            var mins:String = String(dateObj.getMinutes());
            if(mins.length == 1)
                mins = "0" + mins;

            var secs:String = String(dateObj.getSeconds());
            if(secs.length == 1)
                secs = "0" + secs;

            var ms:String = String(dateObj.getMilliseconds());
            if(ms.length == 1)
                ms = "00" + ms;
            else if(ms.length == 2)
                ms = "0" + ms;

            msg = "[" + year + "-" + mon + "-" + day + "_" + hours + ":" + mins + ":" + secs + "." + ms + "] " + msg;

            trace(msg);

            dateObj = null;

            return msg;
        }
    }
}
