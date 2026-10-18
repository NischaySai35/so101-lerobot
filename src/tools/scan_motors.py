"""Read-only ping of Feetech servo IDs 1-10. Usage: python scan_motors.py [port]  (default: follower)"""
import sys
import scservo_sdk as scs

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/so100_follower"
port = scs.PortHandler(PORT)
packet = scs.PacketHandler(0)
assert port.openPort() and port.setBaudRate(1_000_000), f"cannot open {PORT}"
found = []
for i in range(1, 11):
    model, res, err = packet.ping(port, i)
    if res == scs.COMM_SUCCESS:
        found.append(i)
        print(f"ID {i}: model {model}")
port.closePort()
print("found IDs:", found or "none")
