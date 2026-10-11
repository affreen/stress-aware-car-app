"""
BLE Sensor Adapter
------------------
Connects to a standard BLE Heart Rate Service device (Polar H10, Wahoo
Tickr, etc.) and pushes decoded readings to a callback. This is the only
file that knows about Bluetooth -- swapping sensors later (a different
protocol, a camera-based signal, etc.) means changing this file, not the
AI layer or the server.
"""

import asyncio
from typing import Callable, Optional

from bleak import BleakScanner, BleakClient

from ai_layer import SensorReading

HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def _parse_heart_rate(data: bytearray):
    flags = data[0]
    hr_format_16bit = flags & 0x1
    rr_present = flags & 0x10

    index = 1
    if hr_format_16bit:
        bpm = int.from_bytes(data[index:index + 2], byteorder="little")
        index += 2
    else:
        bpm = data[index]
        index += 1

    rr_intervals = []
    if rr_present:
        while index + 1 < len(data):
            rr_raw = int.from_bytes(data[index:index + 2], byteorder="little")
            rr_intervals.append((rr_raw / 1024.0) * 1000.0)
            index += 2

    return bpm, rr_intervals


class BleHeartRateSensor:
    """
    on_reading: callback invoked with a SensorReading each time new data
                arrives from the device.
    on_status:  optional callback invoked with human-readable status
                strings (e.g. "scanning...", "connected", "no device found")
    """

    def __init__(
        self,
        on_reading: Callable[[SensorReading], None],
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.on_reading = on_reading
        self.on_status = on_status or (lambda msg: None)

    def _notify(self, sender, data: bytearray):
        bpm, rr_intervals = _parse_heart_rate(data)
        self.on_reading(SensorReading(bpm=bpm, rr_intervals_ms=rr_intervals))

    async def run(self):
        """
        Scans for a device, connects, and streams readings forever
        (until cancelled). Intended to be run as a background asyncio task.
        """
        self.on_status("Scanning for BLE heart rate devices...")
        devices = await BleakScanner.discover(timeout=10, return_adv=True)

        hr_devices = []
        for device, adv_data in devices.values():
            service_uuids = [u.lower() for u in (adv_data.service_uuids or [])]
            if HEART_RATE_SERVICE_UUID in service_uuids:
                hr_devices.append(device)

        if not hr_devices:
            self.on_status("No BLE heart rate device found. Running in manual/demo mode.")
            return

        selected = hr_devices[0]
        self.on_status(f"Connecting to {selected.name or selected.address}...")

        async with BleakClient(selected.address) as client:
            self.on_status(f"Connected to {selected.name or selected.address}")
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, self._notify)
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                self.on_status("Disconnected.")
