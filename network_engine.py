"""
bREadbeats - Network Engine
TCP connection to restim using T-code format.
Sends alpha/beta position commands.
"""

import socket
import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass

from config import Config


@dataclass
class TCodeCommand:
        """T-code command for restim"""
        alpha: float      # Alpha position (-1.0 to 1.0, will be mapped to 0-9999)
        beta: float       # Beta position (-1.0 to 1.0, will be mapped to 0-9999)
        duration_ms: int  # Duration for the move
        volume: float = 1.0  # Volume (0.0 to 1.0)

        def to_tcode(self) -> str:
                """
                Convert to T-code string for restim.
                restim coordinate system:
                    - L0 = vertical axis (our alpha/Y, negated)
                    - L1 = horizontal axis (our beta/X, negated)
                Rotated 90 degrees clockwise to match restim display orientation
                """
                # Rotate 90 degrees clockwise: swap and negate appropriately
                rotated_alpha = self.beta
                rotated_beta = -self.alpha

                # Map -1.0..1.0 to 0..9999
                l0_val = int((-rotated_alpha + 1.0) / 2.0 * 9999)  # rotated_alpha -> L0 (vertical, negated)
                l1_val = int((-rotated_beta + 1.0) / 2.0 * 9999)   # rotated_beta -> L1 (horizontal, negated)

                # Clamp to valid range
                l0_val = max(0, min(9999, l0_val))
                l1_val = max(0, min(9999, l1_val))

                # Volume to 0..9999
                v0_val = int(max(0.0, min(1.0, self.volume)) * 9999)

                # Build command string - all axes in one message
                # Format: L0xxxxIyyy L1xxxxIyyy V0xxxxIyyy [P0xxxx]
                cmd = f"L0{l0_val:04d}I{self.duration_ms} L1{l1_val:04d}I{self.duration_ms} V0{v0_val:04d}I{self.duration_ms}"
                    # Add P0xxxxIyyy if present (4 digits, 0000-1000)
                p0_val = getattr(self, 'pulse_freq', None)
                if p0_val is not None:
                        cmd += f" P0{int(p0_val):04d}I{self.duration_ms}"
                # Add any other tcode_tags if present
                tcode_tags = getattr(self, 'tcode_tags', {})
                for tag, val in tcode_tags.items():
                    if tag != 'P0':
                        cmd += f" {tag}{int(val):04d}"
                cmd += "\n"
                return cmd


class NetworkEngine:
    """
    Engine 2: The Hands
    Manages TCP connection to restim and sends T-code commands.
    """
    
    def __init__(self, config: Config, 
                 status_callback: Optional[Callable[[str, bool], None]] = None):
        """
        Args:
            config: Application configuration
            status_callback: Called with (status_message, is_connected)
        """
        self.config = config
        self.status_callback = status_callback
        
        # Connection state
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False
        
        # Command queue (thread-safe)
        self.cmd_queue: queue.Queue[TCodeCommand] = queue.Queue()
        
        # Control
        self.sending_enabled = False  # Play/pause control
        
        # Worker thread
        self.worker_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the network engine"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[NetworkEngine] Started")
        
        # Auto-connect if configured
        if self.config.connection.auto_connect:
            self.connect()
            
    def stop(self):
        """Stop the network engine"""
        self.running = False
        self.disconnect()
        
        # Clear queue
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                break
                
        print("[NetworkEngine] Stopped")
        
    def connect(self) -> bool:
        """Connect to restim"""
        if self.connected:
            return True
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout for connect
            self.socket.connect((
                self.config.connection.host,
                self.config.connection.port
            ))
            self.socket.settimeout(1.0)  # 1 second timeout for operations
            
            self.connected = True
            self._was_connected = True  # Track that we've successfully connected before
            self._notify_status(f"Connected to restim at {self.config.connection.host}:{self.config.connection.port}", True)
            print(f"[NetworkEngine] Connected to restim")
            return True
            
        except Exception as e:
            self._notify_status(f"Connection failed: {e}", False)
            print(f"[NetworkEngine] Connection failed: {e}")
            self.socket = None
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from restim"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        self.connected = False
        self._notify_status("Disconnected", False)
        print("[NetworkEngine] Disconnected")
        
    def send_command(self, cmd: TCodeCommand):
        """Queue a command to send"""
        if self.running:
            self.cmd_queue.put(cmd)
            
    def send_test(self):
        """Send a test pattern to verify connection"""
        if not self.connected:
            print("[NetworkEngine] Cannot test - not connected")
            return
            
        print("[NetworkEngine] Sending test pattern...")
        
        # Test pattern with longer durations so each point is visible
        # Our display: alpha=X(horizontal), beta=Y(vertical)
        # restim: L0=vertical, L1=horizontal (we swap in to_tcode)
        # So TCodeCommand(alpha, beta) = (X, Y) in our display
        test_cmds = [
            # Start at center
            ("Center", TCodeCommand(0.0, 0.0, 1000)),
            # Go to each cardinal direction (alpha=X, beta=Y)
            ("Top", TCodeCommand(0.0, 1.0, 1000)),      # Y+
            ("Center", TCodeCommand(0.0, 0.0, 500)),
            ("Bottom", TCodeCommand(0.0, -1.0, 1000)),  # Y-
            ("Center", TCodeCommand(0.0, 0.0, 500)),
            ("Right", TCodeCommand(1.0, 0.0, 1000)),    # X+
            ("Center", TCodeCommand(0.0, 0.0, 500)),
            ("Left", TCodeCommand(-1.0, 0.0, 1000)),    # X-
            ("Center", TCodeCommand(0.0, 0.0, 1000)),
        ]
        
        # Send with real delays between commands
        def send_sequence():
            for name, cmd in test_cmds:
                if not self.connected:
                    break
                print(f"[Test] -> {name} (a={cmd.alpha}, b={cmd.beta})")
                self._send_tcode(cmd)
                time.sleep(cmd.duration_ms / 1000.0)  # Wait for move to complete
            print("[NetworkEngine] Test pattern complete")
        
        # Run in separate thread to not block
        threading.Thread(target=send_sequence, daemon=True).start()
            
    def set_sending_enabled(self, enabled: bool):
        """Enable/disable sending commands (play/pause)"""
        self.sending_enabled = enabled
        status = "Playing" if enabled else "Paused"
        print(f"[NetworkEngine] {status}")
        
    def _worker_loop(self):
        """Background worker that sends queued commands"""
        reconnect_timer = 0
        
        while self.running:
            # Handle reconnection (only if we were connected before and lost connection)
            if not self.connected and self.config.connection.auto_connect and self.socket is None:
                if reconnect_timer <= 0:
                    # Only auto-reconnect if we've been disconnected, not on initial start
                    if hasattr(self, '_was_connected') and self._was_connected:
                        self.connect()
                    reconnect_timer = self.config.connection.reconnect_delay_ms / 1000.0
                else:
                    reconnect_timer -= 0.1
                    
            # Process command queue
            try:
                cmd = self.cmd_queue.get(timeout=0.1)
                
                if self.connected and self.sending_enabled:
                    self._send_tcode(cmd)
                    
                self.cmd_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[NetworkEngine] Worker error: {e}")
                
    def _send_tcode(self, cmd: TCodeCommand):
        """Send a T-code command over the socket"""
        if not self.socket:
            return
            
        tcode = cmd.to_tcode()
        
        try:
            self.socket.sendall(tcode.encode('utf-8'))
            # print(f"[NetworkEngine] Sent: {tcode.strip()}")  # Debug
        except socket.timeout:
            print("[NetworkEngine] Send timeout")
        except Exception as e:
            print(f"[NetworkEngine] Send error: {e}")
            self.disconnect()
            
    def _notify_status(self, message: str, connected: bool):
        """Notify status callback"""
        if self.status_callback:
            self.status_callback(message, connected)


# Test
if __name__ == "__main__":
    from config import Config
    
    def on_status(msg, connected):
        print(f"Status: {msg} (connected={connected})")
        
    config = Config()
    engine = NetworkEngine(config, on_status)
    
    print("Starting network engine...")
    engine.start()
    
    print("\nAttempting to connect...")
    time.sleep(2)
    
    if engine.connected:
        print("\nSending test pattern...")
        engine.set_sending_enabled(True)
        engine.send_test()
        time.sleep(3)
        
    engine.stop()
    print("Done")
