import sounddevice as sd
import numpy as np
import time

# Device 15 = LOGITECH SPEAKERS (WASAPI OUTPUT for loopback)
device = 15

print(f"Testing WASAPI loopback on device {device}")
print("Play some music and watch the RMS values...")
print("Press Ctrl+C to stop\n")

def callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    
    # Calculate RMS
    rms = np.sqrt(np.mean(indata**2))
    print(f"RMS: {rms:.6f}", end='\r')

try:
    # Try with WASAPI settings
    stream = sd.InputStream(
        device=device,
        channels=2,
        samplerate=44100,
        blocksize=1024,
        callback=callback,
        extra_settings=sd.WasapiSettings(exclusive=False)
    )
    
    stream.start()
    print("Stream started successfully!")
    
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped")
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying without WASAPI settings...")
    try:
        stream = sd.InputStream(
            device=device,
            channels=2,
            samplerate=44100,
            blocksize=1024,
            callback=callback
        )
        stream.start()
        print("Stream started!")
        while True:
            time.sleep(0.1)
    except Exception as e2:
        print(f"Also failed: {e2}")
