import sounddevice as sd

print("=== Default Devices ===")
try:
    default_input = sd.query_devices(kind='input')
    print(f"Default INPUT: [{default_input['index']}] {default_input['name']}")
except:
    print("No default input")

try:
    default_output = sd.query_devices(kind='output')
    print(f"Default OUTPUT: [{default_output['index']}] {default_output['name']}")
    print(f"  ^ This is where your audio is playing - use this device for WASAPI loopback!")
except:
    print("No default output")

print("\n=== WASAPI Output Devices (for loopback) ===")
hostapis = sd.query_hostapis()
wasapi_idx = None
for idx, api in enumerate(hostapis):
    if 'WASAPI' in api['name']:
        wasapi_idx = idx
        break

devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['hostapi'] == wasapi_idx and d['max_output_channels'] > 0:
        print(f"[{i}] {d['name']} ({d['max_output_channels']} channels)")
