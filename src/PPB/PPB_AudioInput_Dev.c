#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <ppapi/c/ppp.h>
#include <ppapi/c/ppp_instance.h>
#include <ppapi/c/pp_time.h>
#include <ppapi/c/pp_errors.h>
#include <ppapi/c/pp_completion_callback.h>

#include <ppapi/c/dev/ppb_audio_input_dev.h>

#include "log.h"
#include "res.h"

/**
 * Creates an audio input resource.
 *
 * @param[in] instance A <code>PP_Instance</code> identifying one instance of
 * a module.
 *
 * @return A <code>PP_Resource</code> corresponding to an audio input resource
 * if successful, 0 if failed.
 */
static PP_Resource Create(PP_Instance instance)
{
    LOG("");
    res_add_ref(instance);
    return instance;
};

/**
 * Determines if the given resource is an audio input resource.
 *
 * @param[in] resource A <code>PP_Resource</code> containing a resource.
 *
 * @return A <code>PP_Bool</code> containing <code>PP_TRUE</code> if the given
 * resource is an audio input resource, otherwise <code>PP_FALSE</code>.
 */
static PP_Bool IsAudioInput(PP_Resource resource)
{
    LOG_NP;
    return 0;
};

/**
 * Enumerates audio input devices.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 * @param[in] output An output array which will receive
 * <code>PPB_DeviceRef_Dev</code> resources on success. Please note that the
 * ref count of those resources has already been increased by 1 for the
 * caller.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run on
 * completion.
 *
 * @return An error code from <code>pp_errors.h</code>.
 */
static int32_t EnumerateDevices(PP_Resource audio_input,
    struct PP_ArrayOutput output, struct PP_CompletionCallback callback)
{
    LOG_TD;
    return PP_OK_COMPLETIONPENDING;
};


/**
 * Requests device change notifications.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 * @param[in] callback The callback to receive notifications. If not NULL, it
 * will be called once for the currently available devices, and then every
 * time the list of available devices changes. All calls will happen on the
 * same thread as the one on which MonitorDeviceChange() is called. It will
 * receive notifications until <code>audio_input</code> is destroyed or
 * <code>MonitorDeviceChange()</code> is called to set a new callback for
 * <code>audio_input</code>. You can pass NULL to cancel sending
 * notifications.
 * @param[inout] user_data An opaque pointer that will be passed to
 * <code>callback</code>.
 *
 * @return An error code from <code>pp_errors.h</code>.
 */
static int32_t MonitorDeviceChange(PP_Resource audio_input,
    PP_MonitorDeviceChangeCallback callback, void* user_data)
{
    LOG_NP;
    return 0;
};


/**
 * Opens an audio input device. No sound will be captured until
 * StartCapture() is called.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 * @param[in] device_ref Identifies an audio input device. It could be one of
 * the resource in the array returned by EnumerateDevices(), or 0 which means
 * the default device.
 * @param[in] config A <code>PPB_AudioConfig</code> audio configuration
 * resource.
 * @param[in] audio_input_callback A <code>PPB_AudioInput_Callback</code>
 * function that will be called when data is available.
 * @param[inout] user_data An opaque pointer that will be passed into
 * <code>audio_input_callback</code>.
 * @param[in] callback A <code>PP_CompletionCallback</code> to run when this
 * open operation is completed.
 *
 * @return An error code from <code>pp_errors.h</code>.
 */
static int32_t Open(PP_Resource audio_input, PP_Resource device_ref, PP_Resource config,
    PPB_AudioInput_Callback audio_input_callback, void* user_data,
    struct PP_CompletionCallback callback)
{
    LOG_NP;
    return 0;
};

/**
 * Returns an audio config resource for the given audio input resource.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 *
 * @return A <code>PP_Resource</code> containing the audio config resource if
 * successful.
 */
static PP_Resource GetCurrentConfig(PP_Resource audio_input)
{
    LOG_NP;
    return 0;
};

/**
 * Starts the capture of the audio input resource and begins periodically
 * calling the callback.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 *
 * @return A <code>PP_Bool</code> containing <code>PP_TRUE</code> if
 * successful, otherwise <code>PP_FALSE</code>.
 * Also returns <code>PP_TRUE</code> (and is a no-op) if called while capture
 * is already started.
 */
static PP_Bool StartCapture(PP_Resource audio_input)
{
    LOG_NP;
    return 0;
};

/**
 * Stops the capture of the audio input resource.
 *
 * @param[in] audio_input A PP_Resource containing the audio input resource.
 *
 * @return A <code>PP_Bool</code> containing <code>PP_TRUE</code> if
 * successful, otherwise <code>PP_FALSE</code>.
 * Also returns <code>PP_TRUE</code> (and is a no-op) if called while capture
 * is already stopped. If a buffer is being captured, StopCapture will block
 * until the call completes.
 */
static PP_Bool StopCapture(PP_Resource audio_input)
{
    LOG_NP;
    return 0;
};

/**
 * Closes the audio input device, and stops capturing if necessary. It is
 * not valid to call Open() again after a call to this method.
 * If an audio input resource is destroyed while a device is still open, then
 * it will be implicitly closed, so you are not required to call this method.
 *
 * @param[in] audio_input A <code>PP_Resource</code> corresponding to an audio
 * input resource.
 */
static void Close(PP_Resource audio_input)
{
    LOG_NP;
};

/**
 * The <code>PPB_AudioInput_Dev</code> interface contains pointers to several
 * functions for handling audio input resources.
 *
 * TODO(brettw) before moving out of dev, we need to resolve the issue of
 * the mismatch between the current audio config interface and this one.
 *
 * In particular, the params for input assume stereo, but this class takes
 * everything as mono. We either need to not use an audio config resource, or
 * add mono support.
 *
 * In addition, RecommendSampleFrameCount is completely wrong for audio input.
 * RecommendSampleFrameCount returns the frame count for the current
 * low-latency output device, which is likely inappropriate for a random input
 * device. We may want to move the "recommend" functions to the input or output
 * classes rather than the config.
 */
struct PPB_AudioInput_Dev_0_4 PPB_AudioInput_Dev_0_4_instance =
{
    .Create = Create,
    .IsAudioInput = IsAudioInput,
    .EnumerateDevices = EnumerateDevices,
    .MonitorDeviceChange = MonitorDeviceChange,
    .Open = Open,
    .GetCurrentConfig = GetCurrentConfig,
    .StartCapture = StartCapture,
    .StopCapture = StopCapture,
    .Close = Close,
};
