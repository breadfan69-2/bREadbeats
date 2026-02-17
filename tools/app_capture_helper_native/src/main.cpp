#include "loopback_capture.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

namespace {

std::optional<std::string> getArgValue(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == key) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool hasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            return true;
        }
    }
    return false;
}

int parseIntOrDefault(const std::optional<std::string>& value, int fallback) {
    if (!value.has_value()) {
        return fallback;
    }
    try {
        return std::stoi(*value);
    } catch (...) {
        return fallback;
    }
}

void emitProbeJson(const ProbeResult& result) {
    std::cout
        << "{\"supported\":" << (result.supported ? "true" : "false")
        << ",\"stream_enabled\":" << (result.streamEnabled ? "true" : "false")
        << ",\"reason\":\"" << result.reason << "\"}"
        << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    const bool probeMode = hasFlag(argc, argv, "--probe");
    const bool streamMode = hasFlag(argc, argv, "--stream");

    const int pid = parseIntOrDefault(getArgValue(argc, argv, "--pid"), 0);
    const bool includeChildren = parseIntOrDefault(getArgValue(argc, argv, "--include-children"), 1) == 1;

    if (probeMode) {
        if (pid <= 0) {
            emitProbeJson(ProbeResult{false, false, "invalid pid"});
            return 0;
        }
        emitProbeJson(LoopbackCapture::probeForProcess(pid, includeChildren));
        return 0;
    }

    if (streamMode) {
        const int sampleRate = parseIntOrDefault(getArgValue(argc, argv, "--sample-rate"), 44100);
        const int channels = parseIntOrDefault(getArgValue(argc, argv, "--channels"), 2);
        const int framesPerBuffer = parseIntOrDefault(getArgValue(argc, argv, "--frames-per-buffer"), 1024);

        if (pid <= 0 || sampleRate <= 0 || channels <= 0 || framesPerBuffer <= 0) {
            std::cerr << "invalid stream arguments" << std::endl;
            return 2;
        }

        return LoopbackCapture::streamForProcess(
            pid,
            includeChildren,
            sampleRate,
            channels,
            framesPerBuffer
        );
    }

    std::cerr << "expected --probe or --stream" << std::endl;
    return 2;
}
