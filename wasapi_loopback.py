"""
WASAPI Loopback Recording
Based on: https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
"""

import comtypes  # type: ignore
from comtypes import GUID, CLSCTX_ALL  # type: ignore
from ctypes import POINTER, cast, c_float, c_int32, Structure, c_uint32, c_void_p
from ctypes.wintypes import DWORD, BOOL, WORD, BYTE
import numpy as np
import threading
import time

# Core Audio API GUIDs
CLSID_MMDeviceEnumerator = GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
IID_IMMDeviceEnumerator = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
IID_IAudioClient = GUID('{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}')
IID_IAudioCaptureClient = GUID('{C8ADBD64-E71E-48a0-A4DE-185C395CD317}')

# Audio client activation flags
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000

# Data flow
eRender = 0
eCapture = 1
eAll = 2

# Device role
eConsole = 0
eMultimedia = 1
eCommunications = 2

class WAVEFORMATEX(Structure):
    _fields_ = [
        ('wFormatTag', WORD),
        ('nChannels', WORD),
        ('nSamplesPerSec', DWORD),
        ('nAvgBytesPerSec', DWORD),
        ('nBlockAlign', WORD),
        ('wBitsPerSample', WORD),
        ('cbSize', WORD),
    ]

class WASAPILoopback:
    """Capture system audio using WASAPI loopback mode"""
    
    def __init__(self, callback, buffer_duration_ms=10):
        self.callback = callback
        self.buffer_duration_ms = buffer_duration_ms
        self.running = False
        self.thread = None
        
        # Initialize COM
        import comtypes  # type: ignore
        comtypes.CoInitialize()
        
        # Get device enumerator
        try:
            from comtypes.gen import MMDeviceAPILib as MMDeviceAPI  # type: ignore
        except ImportError:
            # Generate type library
            import comtypes.client  # type: ignore
            comtypes.client.GetModule('mmdevapi.tlb')
            from comtypes.gen import MMDeviceAPILib as MMDeviceAPI  # type: ignore
        
        self.MMDeviceAPI = MMDeviceAPI
        
    def start(self):
        """Start capturing audio"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop capturing audio"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _capture_loop(self):
        """Main capture loop"""
        try:
            # Create device enumerator
            enumerator = comtypes.CoCreateInstance(
                CLSID_MMDeviceEnumerator,
                self.MMDeviceAPI.IMMDeviceEnumerator,
                CLSCTX_ALL
            )
            
            # Get default audio endpoint (speakers/headphones)
            device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)  # type: ignore
            
            # Activate audio client
            audio_client = device.Activate(
                self.MMDeviceAPI.IID_IAudioClient,
                CLSCTX_ALL,
                None
            )
            
            # Get mix format
            wave_format = audio_client.GetMixFormat()
            
            print(f"[WASAPI] Format: {wave_format.nChannels}ch, {wave_format.nSamplesPerSec}Hz, {wave_format.wBitsPerSample}bit")
            
            # Initialize audio client in loopback mode
            buffer_duration = int(self.buffer_duration_ms * 10000)  # Convert to 100-nanosecond units
            
            audio_client.Initialize(
                self.MMDeviceAPI.AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                buffer_duration,
                0,
                wave_format,
                None
            )
            
            # Get capture client
            capture_client = audio_client.GetService(self.MMDeviceAPI.IID_IAudioCaptureClient)
            
            # Get buffer size
            buffer_frame_count = audio_client.GetBufferSize()
            
            # Start capture
            audio_client.Start()
            print("[WASAPI] Loopback capture started")
            
            # Capture loop
            while self.running:
                # Wait for buffer
                time.sleep(self.buffer_duration_ms / 1000.0)
                
                # Get available frames
                next_packet_size = capture_client.GetNextPacketSize()
                
                while next_packet_size > 0:
                    # Get buffer
                    data_pointer, frames_available, flags, device_position, qpc_position = \
                        capture_client.GetBuffer()
                    
                    if frames_available > 0:
                        # Calculate buffer size in bytes
                        bytes_per_frame = wave_format.nChannels * (wave_format.wBitsPerSample // 8)
                        buffer_size = frames_available * bytes_per_frame
                        
                        # Create numpy array from buffer
                        if wave_format.wBitsPerSample == 16:
                            dtype = np.int16
                        elif wave_format.wBitsPerSample == 32:
                            dtype = np.float32
                        else:
                            dtype = np.int16
                        
                        # Copy data
                        audio_data = np.frombuffer(
                            (c_float * (buffer_size // 4)).from_address(data_pointer),
                            dtype=dtype,
                            count=frames_available * wave_format.nChannels
                        ).copy()
                        
                        # Reshape to (frames, channels)
                        audio_data = audio_data.reshape(frames_available, wave_format.nChannels)
                        
                        # Convert to float32 if needed
                        if dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        
                        # Call callback
                        if self.callback:
                            self.callback(audio_data)
                    
                    # Release buffer
                    capture_client.ReleaseBuffer(frames_available)
                    
                    # Check for next packet
                    next_packet_size = capture_client.GetNextPacketSize()
            
            # Stop capture
            audio_client.Stop()
            print("[WASAPI] Loopback capture stopped")
            
        except Exception as e:
            print(f"[WASAPI] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            comtypes.CoUninitialize()


if __name__ == "__main__":
    # Test
    def callback(data):
        rms = np.sqrt(np.mean(data**2))
        bar = '█' * int(rms * 50)
        print(f"RMS: {rms:.6f} {bar}".ljust(70), end='\r')
    
    print("Testing WASAPI loopback...")
    print("Play some music!")
    
    loopback = WASAPILoopback(callback)
    loopback.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        loopback.stop()
