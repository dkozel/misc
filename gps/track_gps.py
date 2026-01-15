#!/usr/bin/env python3
"""
GPS tracker that logs position, heading, and speed to GeoJSON and KML formats.
"""

import sys
import json
import socket
import signal
from datetime import datetime
from pathlib import Path
import geojson
import simplekml


class GPSTracker:
    def __init__(self, output_dir="tracks", host='localhost', port=2947, kml_update_interval=10):
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.geojson_file = self.output_dir / f"track_{timestamp}.geojson"
        self.kml_file = self.output_dir / f"track_{timestamp}.kml"

        # Track data for KML
        self.track_points = []

        # Socket for gpsd
        self.sock = None

        # Running flag
        self.running = True

        # KML update interval (number of points between writes)
        self.kml_update_interval = kml_update_interval
        self.points_since_kml_update = 0

    def connect(self):
        """Connect to gpsd."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))

            # Enable watch mode to receive GPS data
            self.sock.sendall(b'?WATCH={"enable":true,"json":true}\n')
            print(f"Connected to gpsd at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to gpsd: {e}")
            return False

    def process_tpv(self, msg):
        """Process TPV (Time-Position-Velocity) message."""
        # Extract GPS data
        lat = msg.get('lat')
        lon = msg.get('lon')

        # Skip if no position fix
        if lat is None or lon is None:
            return

        # Get optional fields
        altitude = msg.get('alt')
        speed = msg.get('speed')  # m/s
        track = msg.get('track')  # degrees
        time = msg.get('time')

        # Create GeoJSON point feature
        point = geojson.Point((lon, lat, altitude if altitude else 0))

        properties = {
            'time': time,
            'speed': speed,
            'heading': track,
        }

        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        feature = geojson.Feature(geometry=point, properties=properties)

        # Write to newline-delimited GeoJSON
        with open(self.geojson_file, 'a') as f:
            f.write(json.dumps(feature) + '\n')

        # Store for KML track
        track_point = {
            'lon': lon,
            'lat': lat,
            'alt': altitude if altitude else 0,
            'time': time,
            'speed': speed,
            'heading': track,
        }
        self.track_points.append(track_point)
        self.points_since_kml_update += 1

        # Periodically write KML file
        if self.points_since_kml_update >= self.kml_update_interval:
            self.write_kml()
            self.points_since_kml_update = 0

        # Display current status
        speed_kmh = speed * 3.6 if speed is not None else 0
        print(f"Position: {lat:.6f}, {lon:.6f} | "
              f"Speed: {speed_kmh:.1f} km/h | "
              f"Heading: {track:.1f}° | "
              f"Points: {len(self.track_points)}")

    def write_kml(self, verbose=False):
        """Write accumulated track to KML file."""
        if not self.track_points:
            if verbose:
                print("No track points to write")
            return

        kml = simplekml.Kml()

        # Create linestring for the track
        coords = [(p['lon'], p['lat'], p['alt']) for p in self.track_points]
        linestring = kml.newlinestring(name=f"GPS Track {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        linestring.coords = coords
        linestring.style.linestyle.color = simplekml.Color.red
        linestring.style.linestyle.width = 3

        # Add placemarks for start and end points
        if len(self.track_points) >= 1:
            start = self.track_points[0]
            start_point = kml.newpoint(name="Start")
            start_point.coords = [(start['lon'], start['lat'], start['alt'])]
            start_point.style.iconstyle.color = simplekml.Color.green
            start_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/grn-circle.png'

        if len(self.track_points) >= 2:
            end = self.track_points[-1]
            end_point = kml.newpoint(name="End")
            end_point.coords = [(end['lon'], end['lat'], end['alt'])]
            end_point.style.iconstyle.color = simplekml.Color.red
            end_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/red-circle.png'

        # Save KML file
        kml.save(str(self.kml_file))
        if verbose:
            print(f"\nKML track written to: {self.kml_file}")

    def run(self):
        """Main tracking loop."""
        if not self.connect():
            return 1

        print(f"Logging to:")
        print(f"  GeoJSON: {self.geojson_file}")
        print(f"  KML: {self.kml_file}")
        print("\nWaiting for GPS data... (Ctrl+C to stop)\n")

        buffer = b''

        try:
            while self.running:
                try:
                    data = self.sock.recv(4096)
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

                            # Process TPV messages (Time-Position-Velocity)
                            if msg.get('class') == 'TPV':
                                self.process_tpv(msg)

                        except json.JSONDecodeError:
                            continue

                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            print("\n\nStopping tracker...")
        finally:
            self.cleanup()

        return 0

    def cleanup(self):
        """Clean up resources and write final KML."""
        if self.sock:
            self.sock.close()

        # Write final KML with verbose output
        self.write_kml(verbose=True)

        print(f"\nTracking session complete:")
        print(f"  Total points: {len(self.track_points)}")
        print(f"  GeoJSON: {self.geojson_file}")
        print(f"  KML: {self.kml_file}")


def main():
    tracker = GPSTracker()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        tracker.running = False

    signal.signal(signal.SIGINT, signal_handler)

    return tracker.run()


if __name__ == '__main__':
    sys.exit(main())
