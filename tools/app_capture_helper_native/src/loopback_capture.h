#pragma once

#include <cstdint>
#include <string>

struct ProbeResult {
    bool supported{false};
    bool streamEnabled{false};
    std::string reason{"not implemented"};
};

class LoopbackCapture {
public:
    static ProbeResult probeForProcess(std::int32_t pid, bool includeChildren);

    static int streamForProcess(
        std::int32_t pid,
        bool includeChildren,
        std::int32_t sampleRate,
        std::int32_t channels,
        std::int32_t framesPerBuffer
    );
};
