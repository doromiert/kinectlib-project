/**
 * kinect_shim.cpp
 * thin C wrapper around libfreenect2's C++ API
 * exposes plain C functions so ctypes can call them
 */

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <cstdlib>
#include <cstdint>

extern "C" {

struct KinectCtx {
    libfreenect2::Freenect2               freenect2;
    libfreenect2::Freenect2Device        *dev      = nullptr;
    libfreenect2::SyncMultiFrameListener *listener = nullptr;
    libfreenect2::FrameMap               frames;
};

#define SHIM_FRAME_COLOR 0
#define SHIM_FRAME_DEPTH 1
#define SHIM_FRAME_IR    2

KinectCtx *kinect_open() {
    auto *ctx = new KinectCtx();

    if (ctx->freenect2.enumerateDevices() == 0) {
        delete ctx;
        return nullptr;
    }

    ctx->dev = ctx->freenect2.openDefaultDevice();
    if (!ctx->dev) {
        delete ctx;
        return nullptr;
    }

    // now listening to all three streams
    ctx->listener = new libfreenect2::SyncMultiFrameListener(
        libfreenect2::Frame::Color |
        libfreenect2::Frame::Depth |
        libfreenect2::Frame::Ir
    );

    ctx->dev->setColorFrameListener(ctx->listener);
    ctx->dev->setIrAndDepthFrameListener(ctx->listener);
    ctx->dev->start();

    return ctx;
}

void kinect_close(KinectCtx *ctx) {
    if (!ctx) return;
    ctx->dev->stop();
    ctx->dev->close();
    delete ctx->listener;
    delete ctx;
}

int kinect_grab(KinectCtx *ctx, int timeout_ms) {
    if (!ctx) return 0;
    if (!ctx->frames.empty())
        ctx->listener->release(ctx->frames);
    return ctx->listener->waitForNewFrame(ctx->frames, timeout_ms) ? 1 : 0;
}

const uint8_t *kinect_frame_data(KinectCtx *ctx, int frame_type,
                                  int *w, int *h, int *bpp) {
    if (!ctx) return nullptr;

    libfreenect2::Frame *f = nullptr;
    if      (frame_type == SHIM_FRAME_COLOR) f = ctx->frames[libfreenect2::Frame::Color];
    else if (frame_type == SHIM_FRAME_DEPTH) f = ctx->frames[libfreenect2::Frame::Depth];
    else if (frame_type == SHIM_FRAME_IR)    f = ctx->frames[libfreenect2::Frame::Ir];

    if (!f) return nullptr;

    *w   = (int)f->width;
    *h   = (int)f->height;
    *bpp = (int)f->bytes_per_pixel;
    return f->data;
}

} // extern "C"
