import sounddevice as sd

dev = sd.query_devices(16)
print(f"Device 16: {dev['name']}")
print(f"Default sample rate: {dev['default_samplerate']}")
print(f"Input channels: {dev['max_input_channels']}")
