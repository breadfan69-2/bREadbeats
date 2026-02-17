#include "loopback_capture.h"

ProbeResult LoopbackCapture::probeForProcess(std::int32_t /*pid*/, bool /*includeChildren*/) {
    return ProbeResult{
        false,
        false,
        "native app-loopback not implemented yet"
    };
}

int LoopbackCapture::streamForProcess(
    std::int32_t /*pid*/,
    bool /*includeChildren*/,
    std::int32_t /*sampleRate*/,
    std::int32_t /*channels*/,
    std::int32_t /*framesPerBuffer*/
) {
    return 3;
}
