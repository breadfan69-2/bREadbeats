import pyaudiowpatch as pyaudio
import numpy as np
import time

print("Testing WASAPI loopback with pyaudiowpatch...")

p = pyaudio.PyAudio()

# Get default WASAPI loopback device
try:
    # Get default WASAPI info
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
    
    print(f"Using: {default_speakers['name']}")
    print(f"Channels: {default_speakers['maxInputChannels']}")
    print(f"Sample Rate: {int(default_speakers['defaultSampleRate'])}")
    print("\nPlay music and watch the RMS values change!")
    print("Press Ctrl+C to stop\n")
    
    def callback(in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        rms = np.sqrt(np.mean(audio_data**2))
        bar = '█' * int(rms * 100)
        print(f"RMS: {rms:.6f} {bar}".ljust(80), end='\r')
        return (in_data, pyaudio.paContinue)
    
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=default_speakers["maxInputChannels"],
        rate=int(default_speakers["defaultSampleRate"]),
        frames_per_buffer=1024,
        input=True,
        input_device_index=default_speakers["index"],
        stream_callback=callback
    )
    
    stream.start_stream()
    
    while stream.is_active():
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nStopping...")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()
