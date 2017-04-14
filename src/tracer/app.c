#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>


#include "app.h"
#include "app_class.h"
#include "ticker.h"
#include "log.h"

#define ASYNC_DO

int app_run(app_t* app)
{
    return 0;
};

int app_stop(app_t* app)
{
    app->f_exit = 1;

    return 0;
};

static int app_buffer_swap_begin(app_t* app, void** pPtr, size_t* sz)
{
    int idx;

    LOG_T("app->inst=%p", app->inst);

    idx = app_buffer_ref(&app->buf_dev, -1);

    if(idx < 0)
        LOG_E("NO spare/spool buffers");

    *pPtr = (idx < 0)?NULL:app->buf_dev.buf[idx];
    *sz = app->buf_dev.size;

    LOG_T("ptr=%p, idx=%d", *pPtr, idx);

    return (idx < 0)?-ENOMEM:0;
};

int cuda_interlace_frames(unsigned char* src_0, unsigned char* src_1, unsigned char* dst, int stride, int height, cudaStream_t cu_stream);

static int app_buffer_swap_end(app_t* app, void** pbuf)
{
    void* buf = *pbuf;
    int idx = app_buffer_idx(&app->buf_dev, buf);

    LOG_T("buf=%p, idx=%d", buf, idx);

    app_buffer_unref(&app->buf_dev, app->vga[1]);
    app->vga[1] = app->vga[0];
    if(idx >= 0)
        app->vga[0] = idx;
    else
        app_buffer_ref(&app->buf_dev, app->vga[0]);

    app->swaps++;

    LOG_T("vga=[%d,%d]", app->vga[0], app->vga[1]);

    if(app->swaps & 1)
    {
        int64_t t1, t2;
        CUresult e;
        CUcontext cu_ctx_pop;
        NDIlib_video_frame_t ndi_video_frame = { };

        t1 = ticker_now();

        if(CUDA_SUCCESS != (e = cuCtxPushCurrent(app->cu_ctx)))
            LOG_E("cuCtxPushCurrent failed: %s", getCudaDrvErrorString(e));

#ifdef ASYNC_DO
        // wait for syncs
        cudaStreamSynchronize(app->cu_streams[0]);
        cudaStreamSynchronize(app->cu_streams[1]);

        // notify
        t2 = ticker_now();
        LOG_T("cudaStreamSynchronize %d ns", (int)(t2 - t1));
        t1 = t2;
#endif

        // 0
        app_buffer_unref(&app->buf_dev, app->ops[0]);
        app->ops[0] = app->vga[0];
        app_buffer_ref(&app->buf_dev, app->ops[0]);
        // 1
        app_buffer_unref(&app->buf_dev, app->ops[1]);
        app->ops[1] = app->vga[1];
        app_buffer_ref(&app->buf_dev, app->ops[1]);
        // 3
        app_buffer_unref(&app->buf_dev, app->ops[3]);
        app->ops[3] = app->ops[2];
        // 2
        app->ops[2] = app_buffer_ref(&app->buf_dev, -1);
        // 5
        app_buffer_unref(&app->buf_host, app->ops[5]);
        app->ops[5] = app->ops[4];
        // 4
        app->ops[4] = app_buffer_ref(&app->buf_host, -1);
        // 6
        app_buffer_unref(&app->buf_host, app->ops[6]);
        app->ops[6] = app_buffer_ref(&app->buf_host, -1);

#if 1
        // run interlacing
        cuda_interlace_frames
        (
            app->buf_dev.buf[ app->ops[1] ],
            app->buf_dev.buf[ app->ops[0] ],
            app->buf_dev.buf[ app->ops[2] ],
            app->stride, app->inst->height,
            app->cu_streams[0]
        );
#else
        // run localizing
        cudaMemcpyAsync
        (
            app->buf_dev.buf[ app->ops[2] ],
            app->buf_dev.buf[ app->ops[0] ],
            app->size,
            cudaMemcpyDeviceToDevice,
            app->cu_streams[0]
        );
#endif

        // run localizing
        cudaMemcpyAsync
        (
            app->buf_host.buf[ app->ops[4] ],
            app->buf_dev.buf[ app->ops[3] ],
            app->size,
            cudaMemcpyDeviceToHost,
            app->cu_streams[1]
        );

        // send data to ndi
        ndi_video_frame.timecode = NDIlib_send_timecode_synthesize;
        ndi_video_frame.FourCC = NDIlib_FourCC_type_BGRA; //NDIlib_FourCC_type_RGBA
        ndi_video_frame.xres = app->inst->width;
        ndi_video_frame.yres = app->inst->height;
        ndi_video_frame.line_stride_in_bytes = app->stride;
        ndi_video_frame.frame_rate_N = app->inst->fps;
        ndi_video_frame.frame_rate_D = 2;
        ndi_video_frame.frame_format_type = NDIlib_frame_format_type_interleaved;
        ndi_video_frame.picture_aspect_ratio = 16.f / 9.f;
        ndi_video_frame.p_data = app->buf_host.buf[ app->ops[5] ];
//        if(app->ndi_sends > 0)
//            NDIlib_send_send_video_async(app->m_p_ndi_send, NULL);
        NDIlib_send_send_video_async(app->m_p_ndi_send, &ndi_video_frame);
        app->ndi_sends++;

#ifndef ASYNC_DO
        // wait for syncs
        cudaStreamSynchronize(app->cu_streams[0]);
        cudaStreamSynchronize(app->cu_streams[1]);
#endif

        if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
            LOG_E("cuCtxPopCurrent failed: %s", getCudaDrvErrorString(e));
        t2 = ticker_now();

        LOG_T("OPS [%3d%3d%3d%3d%3d%3d%3d] %d ns",
            app->ops[0], app->ops[1], app->ops[2], app->ops[3],
            app->ops[4], app->ops[5], app->ops[6], (int)(t2 - t1));
    };

    return 0;
};

int app_create(app_t** p_app, instance_t* inst)
{
    int i;
    CUresult e;
    CUcontext cu_ctx_pop;
    app_t* app;

    if(!p_app)
        return -EINVAL;

    *p_app = app = (app_t*)calloc(1, sizeof(app_t));

    if(!app)
        return -ENOMEM;

    app->inst = inst;
    app->stride = inst->width * 4;
    app->size = app->stride * inst->height;

    inst->buffer_swap_begin = app_buffer_swap_begin;
    inst->buffer_swap_end = app_buffer_swap_end;
    inst->app_data = app;
    inst->app_class = &app_class_struct;

    pthread_mutex_init(&app->lock, NULL);

    if(CUDA_SUCCESS != (e = cuInit(0)))
    {
        LOG_E("cuInit failed");
        return -ENODATA;
    };

    if(CUDA_SUCCESS != (e = cuCtxCreate(&app->cu_ctx, CU_CTX_BLOCKING_SYNC, app->cu_dev)))
    {
        LOG_E("cuCtxCreate failed: %s", getCudaDrvErrorString(e));
        return -ENODATA;
    };

    app_buffer_init(&app->buf_host, 10, app->size, 1);
    app_buffer_init(&app->buf_dev, 10, app->size, 0);

    app->vga[0] = app_buffer_ref(&app->buf_dev, -1);
    app->vga[1] = app_buffer_ref(&app->buf_dev, -1);

    app->ops[0] = app_buffer_ref(&app->buf_dev, -1);
    app->ops[1] = app_buffer_ref(&app->buf_dev, -1);
    app->ops[2] = app_buffer_ref(&app->buf_dev, -1);
    app->ops[3] = app_buffer_ref(&app->buf_dev, -1);
    app->ops[4] = app_buffer_ref(&app->buf_host, -1);
    app->ops[5] = app_buffer_ref(&app->buf_host, -1);
    app->ops[6] = app_buffer_ref(&app->buf_host, -1);

//    for(i = 0; i < APP_OPS_SIZE; i++) app->ops[i] = -1;

    for(i = 0; i < APP_STRM_SIZE; i++)
        cudaStreamCreate(&app->cu_streams[i]);

    if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
    {
        LOG_E("cuCtxPopCurrent failed: %s", getCudaDrvErrorString(e));
        return -EINVAL;
    };

    // Setup the NDI sender parameters
    NDIlib_send_create_t ndi_send_create_desc;
    ndi_send_create_desc.p_ndi_name = "FLAHS1";
    ndi_send_create_desc.p_groups = NULL;
    ndi_send_create_desc.clock_video = 1;
    ndi_send_create_desc.clock_audio = 0;

    // Create the NDI sender instance
    app->m_p_ndi_send = NDIlib_send_create(&ndi_send_create_desc);
    if(!app->m_p_ndi_send)
        return -EFAULT;

    return 0;
};

int app_destroy(app_t** p_app)
{
    int i;
    CUresult e;
    CUcontext cu_ctx_pop;
    app_t* app;

    if(!p_app)
        return -EINVAL;

    app = *p_app;
    *p_app = NULL;

    if(!app)
        return -EINVAL;

    if(CUDA_SUCCESS != (e = cuCtxPushCurrent(app->cu_ctx)))
        LOG_E("cuCtxPushCurrent failed: %s", getCudaDrvErrorString(e));

    app_buffer_release(&app->buf_host);
    app_buffer_release(&app->buf_dev);

    for(i = 0; i < APP_STRM_SIZE; i++)
        cudaStreamDestroy(app->cu_streams[i]);

    if(CUDA_SUCCESS != (e = cuCtxPopCurrent(&cu_ctx_pop)))
        LOG_E("cuCtxPopCurrent failed: %s", getCudaDrvErrorString(e));

    if(CUDA_SUCCESS != (e = cuCtxDestroy(app->cu_ctx)))
        LOG_E("cuCtxDestroy failed: %s", getCudaDrvErrorString(e));

    pthread_mutex_destroy(&app->lock);

    // Release the NDI sender instance
    if(app->m_p_ndi_send)
    {
        NDIlib_send_destroy(app->m_p_ndi_send);
        app->m_p_ndi_send = NULL;
    };

    free(app);

    return 0;
};
