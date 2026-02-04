import sounddevice as sd
import numpy as np
import time

# Device 16 = Stereo Mix (WASAPI)
device = 16

print(f"Testing Stereo Mix on device {device}")
print("Play some music loudly and watch the RMS values...")
print("They should increase significantly when music plays!")
print("Press Ctrl+C to stop\n")

def callback(indata, frames, time_info, status):
    if status:
        print(f"Status: {status}")
    
    # Calculate RMS
    rms = np.sqrt(np.mean(indata**2))
    # Show with more visibility
    bar_length = int(rms * 1000)
    bar = '█' * min(bar_length, 50)
    print(f"RMS: {rms:.6f} {bar}".ljust(80), end='\r')

try:
    stream = sd.InputStream(
        device=device,
        channels=2,
        samplerate=48000,
        blocksize=1024,
        callback=callback
    )
    
    stream.start()
    print("Stream started! Play music NOW...")
    time.sleep(1)
    print()
    
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped")
except Exception as e:
    print(f"Error: {e}")
