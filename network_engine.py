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
from dataclasses import dataclass, field

from config import Config
from logging_utils import log_event


@dataclass
class TCodeCommand:
    """T-code command for restim"""
    alpha: float      # Alpha position (-1.0 to 1.0, will be mapped to 0-9999)
    beta: float       # Beta position (-1.0 to 1.0, will be mapped to 0-9999)
    duration_ms: int  # Duration for the move
    volume: float = 1.0  # Volume (0.0 to 1.0)
    pulse_freq: Optional[int] = None  # Optional P0 frequency (0-9999)
    pulse_freq_duration: Optional[int] = None  # Optional P0 duration (250ms for smooth averaging)
    tcode_tags: dict = field(default_factory=dict)  # Optional additional T-code tags

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
        l0_val = int((-rotated_alpha + 1.0) / 2.0 * 9999)
        l1_val = int((-rotated_beta + 1.0) / 2.0 * 9999)

        # Clamp to valid range
        l0_val = max(0, min(9999, l0_val))
        l1_val = max(0, min(9999, l1_val))

        # Volume to 0..9999
        v0_val = int(max(0.0, min(1.0, self.volume)) * 9999)

        # Build command string: L0xxxxIyyy L1xxxxIyyy V0xxxxIyyy [P0xxxx] [C0xxxx]
        cmd = f"L0{l0_val:04d}I{self.duration_ms} L1{l1_val:04d}I{self.duration_ms} V0{v0_val:04d}I{self.duration_ms}"
        
        # Add P0xxxxIyyy if present (4 digits, 0000-9999)
        p0_val = getattr(self, 'pulse_freq', None)
        if p0_val is not None:
            p0_dur = getattr(self, 'pulse_freq_duration', None) or self.duration_ms
            cmd += f" P0{int(p0_val):04d}I{p0_dur}"
        
        # Add any other tcode_tags if present (with interpolation time)
        tcode_tags = getattr(self, 'tcode_tags', {})
        for tag, val in tcode_tags.items():
            if tag.endswith('_duration'):
                continue  # Skip duration overrides (consumed below)
            if tag != 'P0':
                # Check for tag-specific duration override (e.g. C0_duration, P1_duration)
                tag_dur = tcode_tags.get(f'{tag}_duration', None)
                dur = tag_dur or self.duration_ms
                cmd += f" {tag}{int(val):04d}I{dur}"
        
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
        self._user_disconnected = False  # True when user explicitly clicked disconnect
        self._dry_run = False  # When True, log but do not send
        
        # Command queue (thread-safe)
        self.cmd_queue: queue.Queue[TCodeCommand] = queue.Queue()
        
        # Control
        self.sending_enabled = False  # Play/pause control
        
        # Worker thread
        self.worker_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the network engine"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        log_event("INFO", "NetworkEngine", "Started")
        
        # Auto-connect if configured
        if self.config.connection.auto_connect:
            self.connect()
            
    def stop(self) -> None:
        """Stop the network engine"""
        self.running = False
        self.disconnect()
        
        # Clear queue
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                break
                
        log_event("INFO", "NetworkEngine", "Stopped")
        
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
            log_event("INFO", "NetworkEngine", "Connected", host=self.config.connection.host, port=self.config.connection.port)
            return True
            
        except Exception as e:
            # Truncate error message for cleaner display
            err_str = str(e)
            if len(err_str) > 40:
                err_str = err_str[:40] + "..."
            self._notify_status(f"Connection failed: {err_str}", False)
            log_event("ERROR", "NetworkEngine", "Connection failed", error=err_str)
            self.socket = None
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """Disconnect from restim"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
        self.connected = False
        self._notify_status("Disconnected", False)
        log_event("INFO", "NetworkEngine", "Disconnected")
        
    def user_disconnect(self) -> None:
        """User explicitly disconnected - do NOT auto-reconnect"""
        self._user_disconnected = True
        self.disconnect()
    
    def user_connect(self) -> None:
        """User explicitly clicked connect - clear disconnect flag and connect"""
        self._user_disconnected = False
        self.connect()
    
    def send_command(self, cmd: TCodeCommand) -> None:
        """Queue a command to send"""
        if self.running:
            self.cmd_queue.put(cmd)
            
    def send_test(self) -> None:
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
            
    def set_sending_enabled(self, enabled: bool) -> None:
        """Enable/disable sending commands (play/pause)"""
        self.sending_enabled = enabled
        status = "Playing" if enabled else "Paused"
        log_event("INFO", "NetworkEngine", status)
    
    def send_immediate(self, cmd: TCodeCommand) -> None:
        """Send a command immediately, bypassing the queue and sending_enabled check.
        Used for shutdown/fade-out commands that must be sent regardless of state."""
        if self.connected:
            self._send_tcode(cmd)

    def set_dry_run(self, enabled: bool) -> None:
        """Enable/disable dry-run mode (log only, no network send)."""
        self._dry_run = enabled
        state = "ON" if enabled else "OFF"
        log_event("INFO", "NetworkEngine", f"Dry-run {state}")
        
    def _worker_loop(self) -> None:
        """Background worker that sends queued commands"""
        reconnect_timer = 0
        
        while self.running:
            # Handle reconnection (only if not user-disconnected and we were connected before)
            if not self.connected and self.config.connection.auto_connect and self.socket is None:
                if not self._user_disconnected:
                    if reconnect_timer <= 0:
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
                log_event("ERROR", "NetworkEngine", "Worker error", error=e)
                
    def _send_tcode(self, cmd: TCodeCommand) -> None:
        """Send a T-code command over the socket"""
        if self._dry_run:
            log_event("INFO", "NetworkEngine", "Dry-run", tcode=cmd.to_tcode().strip())
            return

        if not self.socket:
            return
            
        tcode = cmd.to_tcode()
        
        try:
            self.socket.sendall(tcode.encode('utf-8'))
            # print(f"[NetworkEngine] Sent: {tcode.strip()}")  # Debug
        except socket.timeout:
            log_event("WARN", "NetworkEngine", "Send timeout")
        except Exception as e:
            log_event("ERROR", "NetworkEngine", "Send error", error=e)
            self.disconnect()
            
    def _notify_status(self, message: str, connected: bool) -> None:
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
