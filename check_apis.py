import sounddevice as sd

devices = sd.query_devices()
hostapis = sd.query_hostapis()

# Find WASAPI hostapi index
wasapi_idx = None
for idx, api in enumerate(hostapis):
    if 'WASAPI' in api['name']:
        wasapi_idx = idx
        print(f"WASAPI host API index: {idx}")
        break

print("\n=== All Devices ===")
for i, d in enumerate(devices):
    api_name = hostapis[d['hostapi']]['name']
    print(f"[{i}] {d['name']}")
    print(f"    API: {api_name}, Inputs: {d['max_input_channels']}, Outputs: {d['max_output_channels']}")

print("\n=== WASAPI Devices (Loopback Candidates) ===")
for i, d in enumerate(devices):
    if d['hostapi'] == wasapi_idx:
        print(f"[{i}] {d['name']}")
        print(f"    Inputs: {d['max_input_channels']}, Outputs: {d['max_output_channels']}")
        if d['max_output_channels'] > 0:
            print(f"    ⭐ This OUTPUT device can be used as WASAPI loopback")

