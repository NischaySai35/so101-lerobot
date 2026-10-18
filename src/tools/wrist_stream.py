"""Share the robot's gripper camera as a live MJPEG stream on the local network.

Usage:  source scripts/env.sh && python src/tools/wrist_stream.py [--port 8081] [--fps 15]
Viewers open http://<this laptop's IP>:<port>/ in a browser (same Wi-Fi / network). The page has a Snapshot
button (keeps a strip of shots, click one to save it) and Download; /snapshot gives one full-quality JPEG.

This program holds the camera while it runs; the chess code then reads frames from this stream instead
(chess_player/vision/camera.py falls back to http://localhost:<port>/stream when the device is busy).
"""
import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"

latest = {"jpg": None, "frame": None, "t": 0.0}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot camera</title><style>
body{margin:0;background:#0b0d11;color:#e8eaf0;font:14px system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid #252a36}
b{letter-spacing:1px} .sp{flex:1} button,a.btn{font:inherit;color:#1b130a;background:linear-gradient(135deg,#c8a46a,#a9844f);
border:0;border-radius:10px;padding:8px 14px;font-weight:700;cursor:pointer;text-decoration:none}
a.ghost{background:#191d26;color:#e8eaf0;border:1px solid #252a36;font-weight:500}
main{flex:1;display:grid;place-items:center;min-height:0;padding:10px} img{max-width:100%;max-height:100%;border-radius:12px}
#shots{display:flex;gap:8px;padding:8px 14px;overflow-x:auto;border-top:1px solid #252a36;min-height:96px}
#shots a img{height:80px;border-radius:8px;border:1px solid #252a36} .flash{animation:f .3s} @keyframes f{from{filter:brightness(3)}}
</style></head><body>
<header><b>&#9679; ROBOT CAMERA &middot; LIVE</b><span class="sp"></span>
<button id="snap">&#128248; Snapshot</button><a class="btn ghost" href="/snapshot?download=1">&#11015; Download</a></header>
<main><img id="live" src="/stream" alt="live"></main>
<div id="shots"></div>
<script>
document.getElementById('snap').onclick=async()=>{
  const r=await fetch('/snapshot?t='+Date.now()); if(!r.ok) return alert('no picture yet');
  const url=URL.createObjectURL(await r.blob()), name='robot_camera_'+new Date().toISOString().replace(/[:.]/g,'-')+'.jpg';
  const a=document.createElement('a'); a.href=url; a.download=name; a.innerHTML='<img src="'+url+'" title="'+name+' (click to save)">';
  document.getElementById('shots').prepend(a); const l=document.getElementById('live'); l.classList.remove('flash'); void l.offsetWidth; l.classList.add('flash');
};
</script></body></html>"""
lock = threading.Condition()


def capture(fps):
    while True:
        cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                latest["jpg"], latest["frame"], latest["t"] = jpg.tobytes(), frame, time.time()
                lock.notify_all()
        cap.release()
        print("camera lost, reconnecting...")
        time.sleep(1.0)


class Handler(BaseHTTPRequestHandler):
    fps = 15

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/snapshot"):  # one full-quality JPEG; ?download=1 saves it as a file
            with lock:
                frame = latest["frame"]
            if frame is None:
                self.send_error(503, "no frame yet")
                return
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            jpg = jpg.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            if "download=1" in self.path:
                name = time.strftime("robot_camera_%Y%m%d_%H%M%S.jpg")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = 0.0
            try:
                while True:
                    with lock:
                        lock.wait_for(lambda: latest["t"] > last, timeout=2.0)
                        jpg, last = latest["jpg"], latest["t"]
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1.0 / self.fps)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.101.104", 80))  # any LAN address; nothing is sent
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()
    Handler.fps = a.fps
    threading.Thread(target=capture, args=(a.fps,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"Robot camera live at  http://{lan_ip()}:{a.port}/   (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
