#!/usr/bin/env python3
"""
Utility to find USB GPS devices and verify gpsd availability.
"""

import sys
import socket
import subprocess
import json
from pathlib import Path

try:
    import gps
    HAS_GPS_MODULE = True
except ImportError:
    HAS_GPS_MODULE = False


def check_gpsd_running():
    """Check if gpsd service is running."""
    try:
        result = subprocess.run(
            ['systemctl', 'is-active', 'gpsd'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def find_usb_serial_devices():
    """Find potential USB GPS devices."""
    devices = []

    # Common device paths for USB GPS units
    device_patterns = [
        '/dev/ttyUSB*',
        '/dev/ttyACM*',
        '/dev/serial/by-id/usb-*'
    ]

    for pattern in device_patterns:
        devices.extend(Path('/').glob(pattern.lstrip('/')))

    return [str(d) for d in devices if d.exists()]


def check_gpsd_connection_raw(host='localhost', port=2947, timeout=5):
    """Attempt to connect to gpsd using raw socket and JSON protocol."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # Send ?WATCH command to start receiving data
        sock.sendall(b'?WATCH={"enable":true,"json":true}\n')

        # Read responses
        buffer = b''
        devices = []
        start_time = None

        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                data = sock.recv(4096)
                if not data:
                    break

                buffer += data
                lines = buffer.split(b'\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        if msg.get('class') == 'DEVICES':
                            devices = msg.get('devices', [])
                            sock.close()
                            return True, devices
                    except json.JSONDecodeError:
                        continue
            except socket.timeout:
                break

        sock.close()
        # Connected but didn't get device info yet
        return True, devices

    except (ConnectionRefusedError, socket.error) as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_gpsd_connection(host='localhost', port=2947):
    """Attempt to connect to gpsd and get device information."""
    # Try raw socket method first (more reliable across Python versions)
    return check_gpsd_connection_raw(host, port)

    # Legacy gps module method (commented out due to Python 3.12 compatibility issues)
    # if not HAS_GPS_MODULE:
    #     return False, "gps module not available"
    #
    # try:
    #     session = gps.gps(host=host, port=port, mode=gps.WATCH_ENABLE)
    #
    #     for _ in range(5):
    #         report = session.next()
    #         if report['class'] == 'DEVICES':
    #             devices = report.get('devices', [])
    #             session.close()
    #             return True, devices
    #
    #     session.close()
    #     return True, []
    # except Exception as e:
    #     return False, str(e)


def main():
    """Main function to find GPS and check gpsd."""
    print("=" * 60)
    print("USB GPS Finder and gpsd Checker")
    print("=" * 60)

    # Check for USB serial devices
    print("\n1. Checking for USB serial devices...")
    usb_devices = find_usb_serial_devices()

    if usb_devices:
        print(f"   Found {len(usb_devices)} USB serial device(s):")
        for device in usb_devices:
            print(f"   - {device}")
    else:
        print("   No USB serial devices found.")

    # Check if gpsd is running
    print("\n2. Checking gpsd service status...")
    gpsd_running = check_gpsd_running()

    if gpsd_running:
        print("   ✓ gpsd service is active")
    else:
        print("   ✗ gpsd service is not running")
        print("   Hint: Start with 'sudo systemctl start gpsd'")

    # Try to connect to gpsd
    print("\n3. Attempting to connect to gpsd...")
    connected, result = check_gpsd_connection()

    if connected:
        print("   ✓ Successfully connected to gpsd")

        if isinstance(result, list) and result:
            print(f"\n   GPS Devices reported by gpsd ({len(result)}):")
            for idx, device in enumerate(result, 1):
                path = device.get('path', 'Unknown')
                driver = device.get('driver', 'Unknown')
                activated = device.get('activated', 'Unknown')
                print(f"   Device {idx}:")
                print(f"     Path: {path}")
                print(f"     Driver: {driver}")
                print(f"     Activated: {activated}")
        elif isinstance(result, list):
            print("   No GPS devices reported by gpsd yet")
            print("   (This is normal if GPS just started)")
        else:
            print(f"   Connected but couldn't get device info: {result}")
    else:
        print(f"   ✗ Failed to connect to gpsd: {result}")
        print("   Hint: Ensure gpsd is running and configured correctly")

    print("\n" + "=" * 60)

    # Exit code based on success
    if connected and (isinstance(result, list) and len(result) > 0):
        print("Status: GPS found and available via gpsd")
        return 0
    elif connected:
        print("Status: gpsd running but no GPS devices detected yet")
        return 1
    else:
        print("Status: Cannot connect to gpsd")
        return 2


if __name__ == '__main__':
    sys.exit(main())
