#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_video_capture_dev.h>

#include "log.h"
#include "res.h"

/**
 * Creates a new VideoCapture.
 */
static PP_Resource Create(PP_Instance instance)
{
    res_add_ref(instance);
    return instance;
};


/**
 * Returns PP_TRUE if the given resource is a VideoCapture.
 */
static PP_Bool IsVideoCapture(PP_Resource video_capture)
{
    LOG_NP;
    return 0;
};


/**
 * Enumerates video capture devices.
 *
 * @param[in] video_capture A <code>PP_Resource</code> corresponding to a
 * video capture resource.
 * @param[in] output An output array which will receive
 * <code>PPB_DeviceRef_Dev</code> resources on success. Please note that the
 * ref count of those resources has already been increased by 1 for the
 * caller.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * completion.
 *
 * @return An error code from <code>pp_errors.h</code>.
 */
static int32_t EnumerateDevices(PP_Resource video_capture,
    struct PP_ArrayOutput output, struct PP_CompletionCallback callback)
{
    LOG_TD;
    return PP_OK_COMPLETIONPENDING;
};

/**
 * Requests device change notifications.
 *
 * @param[in] video_capture A <code>PP_Resource</code> corresponding to a
 * video capture resource.
 * @param[in] callback The callback to receive notifications. If not NULL, it
 * will be called once for the currently available devices, and then every
 * time the list of available devices changes. All calls will happen on the
 * same thread as the one on which MonitorDeviceChange() is called. It will
 * receive notifications until <code>video_capture</code> is destroyed or
 * <code>MonitorDeviceChange()</code> is called to set a new callback for
 * <code>video_capture</code>. You can pass NULL to cancel sending
 * notifications.
 * @param[inout] user_data An opaque pointer that will be passed to
 * <code>callback</code>.
 *
 * @return An error code from <code>pp_errors.h</code>.
 */
static int32_t MonitorDeviceChange(PP_Resource video_capture,
    PP_MonitorDeviceChangeCallback callback, void* user_data)
{
    LOG_NP;
    return 0;
};


/**
 * Opens a video capture device. |device_ref| identifies a video capture
 * device. It could be one of the resource in the array returned by
 * |EnumerateDevices()|, or 0 which means the default device.
 * |requested_info| is a pointer to a structure containing the requested
 * resolution and frame rate. |buffer_count| is the number of buffers
 * requested by the plugin. Note: it is only used as advisory, the browser may
 * allocate more or fewer based on available resources. How many buffers
 * depends on usage. At least 2 to make sure latency doesn't cause lost
 * frames. If the plugin expects to hold on to more than one buffer at a time
 * (e.g. to do multi-frame processing, like video encoding), it should request
 * that many more.
 */
static int32_t Open(PP_Resource video_capture,
    PP_Resource device_ref, const struct PP_VideoCaptureDeviceInfo_Dev* requested_info,
    uint32_t buffer_count, struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Starts the capture.
 *
 * Returns PP_ERROR_FAILED if called when the capture was already started, or
 * PP_OK on success.
 */
static int32_t StartCapture(PP_Resource video_capture)
{
    LOG_NP;
    return 0;
};


/**
 * Allows the browser to reuse a buffer that was previously sent by
 * PPP_VideoCapture_Dev.OnBufferReady. |buffer| is the index of the buffer in
 * the array returned by PPP_VideoCapture_Dev.OnDeviceInfo.
 *
 * Returns PP_ERROR_BADARGUMENT if buffer is out of range (greater than the
 * number of buffers returned by PPP_VideoCapture_Dev.OnDeviceInfo), or if it
 * is not currently owned by the plugin. Returns PP_OK otherwise.
 */
static int32_t ReuseBuffer(PP_Resource video_capture, uint32_t buffer)
{
    LOG_NP;
    return 0;
};

/**
 * Stops the capture.
 *
 * Returns PP_ERROR_FAILED if the capture wasn't already started, or PP_OK on
 * success.
 */
static int32_t StopCapture(PP_Resource video_capture)
{
    LOG_NP;
    return 0;
};

/**
 * Closes the video capture device, and stops capturing if necessary. It is
 * not valid to call |Open()| again after a call to this method.
 * If a video capture resource is destroyed while a device is still open, then
 * it will be implicitly closed, so you are not required to call this method.
 */
static void Close(PP_Resource video_capture)
{
    LOG_NP;
};

/**
 * Video capture interface. It goes hand-in-hand with PPP_VideoCapture_Dev.
 *
 * Theory of operation:
 * 1- Create a VideoCapture resource using Create.
 * 2- Find available video capture devices using EnumerateDevices.
 * 3- Open a video capture device. In addition to a device reference (0 can be
 * used to indicate the default device), you pass in the requested info
 * (resolution, frame rate), as well as suggest a number of buffers you will
 * need.
 * 4- Start the capture using StartCapture.
 * 5- Receive the OnDeviceInfo callback, in PPP_VideoCapture_Dev, which will
 * give you the actual capture info (the requested one is not guaranteed), as
 * well as an array of buffers allocated by the browser.
 * 6- On every frame captured by the browser, OnBufferReady (in
 * PPP_VideoCapture_Dev) is called with the index of the buffer from the array
 * containing the new frame. The buffer is now "owned" by the plugin, and the
 * browser won't reuse it until ReuseBuffer is called.
 * 7- When the plugin is done with the buffer, call ReuseBuffer.
 * 8- Stop the capture using StopCapture.
 * 9- Close the device.
 *
 * The browser may change the resolution based on the constraints of the system,
 * in which case OnDeviceInfo will be called again, with new buffers.
 *
 * The buffers contain the pixel data for a frame. The format is planar YUV
 * 4:2:0, one byte per pixel, tightly packed (width x height Y values, then
 * width/2 x height/2 U values, then width/2 x height/2 V values).
 */
struct PPB_VideoCapture_Dev_0_3 PPB_VideoCapture_Dev_0_3_instance =
{
    .Create = Create,
    .IsVideoCapture = IsVideoCapture,
    .EnumerateDevices = EnumerateDevices,
    .MonitorDeviceChange = MonitorDeviceChange,
    .Open = Open,
    .StartCapture = StartCapture,
    .ReuseBuffer = ReuseBuffer,
    .StopCapture = StopCapture,
    .Close = Close,
};
