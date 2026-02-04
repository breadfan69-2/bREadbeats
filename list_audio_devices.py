#!/usr/bin/env python3
"""List all audio devices"""
import sounddevice as sd

print("Available Audio Devices:\n")
devs = sd.query_devices()
for i, d in enumerate(devs):
    in_ch = d['max_input_channels']
    out_ch = d['max_output_channels']
    sr = d['default_samplerate']
    print(f"[{i}] {d['name']}")
    print(f"    Input: {in_ch} channels, Output: {out_ch} channels")
    print(f"    Default SR: {sr} Hz")
    print()
