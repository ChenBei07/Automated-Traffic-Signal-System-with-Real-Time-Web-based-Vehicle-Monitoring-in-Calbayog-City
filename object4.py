import io
import logging
import socketserver
from http import server
import cv2
import torch
import numpy as np
import base64
from http.cookies import SimpleCookie
from time import time, strftime, localtime
from pathlib import Path
from picamera2 import Picamera2
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from trafficDB import update_vehicle_count, get_vehicle_counts, get_all_vehicle_data
from history import handle_history_request
from login import check_auth, handle_login_request, handle_logout_request
from userMgt import handle_user_management_request
from userLogs import handle_user_logs_request

PAGE = """
<html>
<head>
    <title>Raspberry Pi - Vehicle Detection</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            text-align: center;
            padding: 8px;
        }
    </style>
</head>
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #2c3e50;
        margin: 0;
        padding: 20px;
        text-align: center;
        color: white;
    }
    button {
        background-color: #27ae60;
        color: white;
        border: none;
        padding: 10px 20px;
        margin: 10px;
        cursor: pointer;
        border-radius: 5px;
        font-size: 16px;
    }
    button:hover {
        background-color: #c0392b;
    }
    h2 {
        color: #f1c40f;
    }
    table {
        border-collapse: collapse;
        width: 90%;
        margin: 20px auto;
        background: white;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        border-radius: 5px;
        overflow: hidden;
        color: black;
    }
    th, td {
        text-align: center;
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    th {
        cursor: pointer;
        background-color: #f39c12;
        color: white;
    }
    th:hover {
        background-color: #e67e22;
    }
    tr:nth-child(even) {
        background-color: #ecf0f1;
    }
    select {
        padding: 8px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        margin: 10px;
    }
    select:focus {
        outline: none;
        border-color: #27ae60;
    }
    .logout-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        padding: 10px 20px;
        background-color: red;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 16px;
        border-radius: 5px;
    }
</style>
<body>
    <button class="logout-btn" onclick="logout()">Logout</button>
    <center>
        <h1>TRACKING</h1>
        <h2 id="datetime">Loading...</h2>
    </center>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div>
            <img src="stream.mjpg" width="640" height="480">
        </div>
        <div id="count-div" style="width: 100%; overflow-x: auto; white-space: nowrap; display: flex; justify-content: center;"></div>
    </div>
    <center>
        <button onclick="window.location.href='/history'">Show History</button>
        <button id="userMgtBtn">User Management</button>
    </center>
    <script>
        var userRole = '';

        function refreshCounts() {
            fetch('/counts')
                .then(response => response.text())
                .then(html => {
                    document.getElementById('count-div').innerHTML = html;
                });
        }

        function updateDateTime() {
            const now = new Date();
            document.getElementById('datetime').innerText = now.toLocaleString();
        }

        function openCenteredPopup(url, title, width, height) {
            const screenWidth = window.screen.width;
            const screenHeight = window.screen.height;
            const left = (screenWidth - width) / 2;
            const top = (screenHeight - height) / 2;
            const features = `width=${width},height=${height},left=${left},top=${top},resizable=no,scrollbars=no`;

            window.open(url, title, features);
        }
        
        function logout() {
            fetch('/logout', { method: 'GET' })
                .then(response => {
                    window.location.href = '/login'; // Redirect to login page
                })
                .catch(error => console.error('Logout failed:', error));
        }

        // Listen for logout event in all tabs/windows
        window.onload = function () {
            if (localStorage.getItem("logout") === "true") {
                localStorage.removeItem("logout"); // Reset flag
                window.location.href = '/login'; // Redirect to login
            }
        };
        
        
        // Call function when button is clicked
        document.getElementById("userMgtBtn").addEventListener("click", function() {
            openCenteredPopup('/create_user', 'CreateUser', 400, 1000);
        });
        
        if (userRole !== "admin") {
            document.getElementById("userMgtBtn").style.display = "none";
        }
        
        setInterval(refreshCounts, 2000);
        setInterval(updateDateTime, 1000);
        updateDateTime();
    </script>
</body>
</html>
"""

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

model_path = Path("yolov5n.pt")
device = select_device('cpu')
model = DetectMultiBackend(model_path, device=device, dnn=False)
model.eval()
model_names = model.names

trackers = []
valid_vehicle_classes = {"car", "bus", "truck", "motorcycle"}

def detect_objects(frame):
    global trackers
    current_hour = int(strftime('%H', localtime()))
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (640, 480))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, 0.20, 0.45, classes=[2, 3, 5, 7], agnostic=False)

    new_trackers = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                class_name = model.names[int(cls)]
                if class_name in valid_vehicle_classes:
                    box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1]))

                    is_new_object = True
                    for tracker, _, _ in trackers:
                        success, tracked_box = tracker.update(img)
                        if success:
                            iou = calculate_iou(box, tracked_box)
                            if iou > 0.20:
                                is_new_object = False
                                break
                    
                    if is_new_object:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(img, box)
                        new_trackers.append((tracker, class_name, conf))
                        update_vehicle_count(class_name, current_hour)
                    else:
                        new_trackers.append((tracker, class_name, conf))

    trackers = new_trackers
    return img

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/login":
            handle_login_request(self)
        elif self.path == "/create_user":
            handle_user_management_request(self)
        else:
            self.send_error(405, "Method Not Allowed")
    
    def do_GET(self): 
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/login')
            self.end_headers()
        elif self.path == '/login':
            handle_login_request(self)
        elif self.path == "/logout":
            handle_logout_request(self)
        elif self.path == '/create_user':
            handle_user_management_request(self)
        elif self.path == "/userLogs":
            handle_user_logs_request(self)
        elif self.path == '/index.html' or self.path == '/history':
            is_authenticated, role = check_auth(self)
            if not is_authenticated:
                self.send_response(303)
                self.send_header('Location', '/login')
                self.end_headers()
                return
            modified_page = PAGE.replace("var userRole = '';", f'var userRole = "{role}";')
            content = modified_page.encode('utf-8')
            if self.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
            elif self.path == '/history':
                handle_history_request(self)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    start_time = time()
                    frame = picam2.capture_array()
                    frame = detect_objects(frame)

                    for tracker, class_name, conf in trackers:
                        success, box = tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in box]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f'{class_name} {conf:.2f}'
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        elif self.path == '/counts':
            results = get_vehicle_counts()
            count_table = "<center><h2>Vehicle Counts (24h)</h2><table border='1'><tr><th>Class/Hour</th>"
            count_table += ''.join([f'<th>{h}:00</th>' for h in range(24)]) + "</tr>"
            
            for cls in valid_vehicle_classes:
                counts = next((row[1:] for row in results if row[0] == cls), [0] * 24)
                count_table += f'<tr><td>{cls}</td>'
                count_table += ''.join([f'<td>{count}</td>' for count in counts]) + '</tr>'
            
            count_table += "</table></center>"
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(count_table.encode('utf-8'))
    
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

address = ('', 8000)
server = StreamingServer(address, StreamingHandler)
print("Streaming on http://192.168.101.13:8000")
server.serve_forever()

picam2.stop()
