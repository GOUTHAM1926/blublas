#include "mycublas.h"
#include <iostream>

// Internal context definition handled in header

extern "C" {

mycublasStatus_t mycublasCreate(mycublasHandle_t *handle) {
    if (handle == nullptr) {
        return MYCUBLAS_STATUS_INVALID_VALUE;
    }

    try {
        struct mycublasContext *ctx = new struct mycublasContext;
        ctx->deviceId = 0; // Default to device 0 for now
        ctx->stream = 0;   // Default stream
        *handle = ctx;
        return MYCUBLAS_STATUS_SUCCESS;
    } catch (...) {
        return MYCUBLAS_STATUS_ALLOC_FAILED;
    }
}

mycublasStatus_t mycublasDestroy(mycublasHandle_t handle) {
    if (handle != nullptr) {
        delete handle;
        return MYCUBLAS_STATUS_SUCCESS;
    }
    return MYCUBLAS_STATUS_NOT_INITIALIZED;
}

mycublasStatus_t mycublasSetStream(mycublasHandle_t handle, cudaStream_t streamId) {
    if (handle == nullptr) {
        return MYCUBLAS_STATUS_INVALID_VALUE;
    }
    handle->stream = streamId;
    return MYCUBLAS_STATUS_SUCCESS;
}

}
