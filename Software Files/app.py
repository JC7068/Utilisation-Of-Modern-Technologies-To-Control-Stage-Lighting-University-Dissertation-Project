from flask import Flask, render_template_string, Response, request, jsonify
import cv2
import numpy as np
import threading
from datetime import datetime
import colorsys
import sacn
import atexit
import json
import os

app = Flask(__name__)

camera = cv2.VideoCapture(0)

click_data = []
click_counter = 0
lock = threading.Lock()

CLICK_DURATION = 30
HEAT_RADIUS = 80
GAUSSIAN_BLUR = 61

beam_x = None
beam_y = None

beam_color = [0, 255, 255]
beam_brightness = 1.0

# =============================
# sACN SETUP
# =============================

sacn_sender = sacn.sACNsender()
sacn_sender.start()
sacn_sender.activate_output(1)
sacn_sender[1].multicast = False
sacn_sender[1].destination = "192.168.137.66"

atexit.register(sacn_sender.stop)

# =============================
# ZONE -> FIXTURE MAPPING
# =============================

SLIMPAR_GRID = [
    [2, 4, 8, 6],
    [1, 3, 7, 5],
]

def slimpar_base(fixture_num):
    return (fixture_num - 1) * 3

COLORRAIL_BASES = [30, 56, 82]

# =============================
# DMX CHANNEL DEFINITIONS
# =============================

# Full descriptive map of every DMX channel being used.
# Each entry: (light_number, dmx_channel, model, feature)
DMX_CHANNEL_MAP = []

# SlimPARs 1-10 (3 channels each: R, G, B)
SLIMPAR_FEATURES = ["Red Brightness", "Green Brightness", "Blue Brightness"]
for fixture in range(1, 11):
    ch_start = (fixture - 1) * 3 + 1
    for i, feature in enumerate(SLIMPAR_FEATURES):
        DMX_CHANNEL_MAP.append({
            "Light Number": fixture,
            "DMX Channel": ch_start + i,
            "Light Model": "Chauvet SlimPAR 56 ILS",
            "Feature": feature,
            "Current Value": 0
        })

# ColorRails 11, 12, 13
# Each: 1x Master Brightness + 8x RGB segments + 1x Strobe = 26 channels
COLORRAIL_STARTS = [31, 57, 83]
for cr_index, cr_start in enumerate(COLORRAIL_STARTS):
    light_num = 11 + cr_index
    DMX_CHANNEL_MAP.append({
        "Light Number": light_num,
        "DMX Channel": cr_start,
        "Light Model": "Chauvet ColorRail IRC",
        "Feature": "Master Brightness",
        "Current Value": 0
    })
    for seg in range(8):
        seg_ch = cr_start + 1 + seg * 3
        for i, feature in enumerate(SLIMPAR_FEATURES):
            DMX_CHANNEL_MAP.append({
                "Light Number": light_num,
                "DMX Channel": seg_ch + i,
                "Light Model": "Chauvet ColorRail IRC",
                "Feature": f"Segment {seg + 1} {feature}",
                "Current Value": 0
            })
    DMX_CHANNEL_MAP.append({
        "Light Number": light_num,
        "DMX Channel": cr_start + 25,
        "Light Model": "Chauvet ColorRail IRC",
        "Feature": "Strobe Effect",
        "Current Value": 0
    })

# Moving Heads 14 and 15
PANTHER_FEATURES = [
    "Pan", "Pan Fine", "Tilt", "Tilt Fine",
    "Color", "Gobo", "Shutter", "Master Brightness",
    "Pan/Tilt Speed", "Function", "Dim Mode"
]
PANTHER_STARTS = [109, 120]
for ph_index, ph_start in enumerate(PANTHER_STARTS):
    light_num = 14 + ph_index
    for i, feature in enumerate(PANTHER_FEATURES):
        DMX_CHANNEL_MAP.append({
            "Light Number": light_num,
            "DMX Channel": ph_start + i,
            "Light Model": "Beamz Panther 40 LED Spot",
            "Feature": feature,
            "Current Value": 0
        })

# Build a quick lookup: dmx channel (1-based) -> index in DMX_CHANNEL_MAP
CHANNEL_LOOKUP = {entry["DMX Channel"]: idx for idx, entry in enumerate(DMX_CHANNEL_MAP)}

# Path for the live JSON output file (same folder as app.py)
DMX_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmx_live.json")

def update_dmx_json(dmx_list):
    """Update DMX_CHANNEL_MAP values from the current dmx list and write to file."""
    for ch_1based, idx in CHANNEL_LOOKUP.items():
        DMX_CHANNEL_MAP[idx]["Current Value"] = dmx_list[ch_1based - 1]
    with open(DMX_JSON_PATH, "w") as f:
        json.dump(DMX_CHANNEL_MAP, f, indent=4)

# =============================
# DMX HELPERS
# =============================

def get_frame_dimensions():
    success, frame = camera.read()
    if success:
        h, w, _ = frame.shape
        return w, h
    return 640, 480

def get_active_zones(bx, by, frame_w, frame_h):
    if bx is None or by is None:
        return None, None, False
    col = min(int((bx / frame_w) * 4), 3)
    row = 0 if by < frame_h / 2 else 1
    active_slimpar = SLIMPAR_GRID[row][col]
    cr_col = min(int((bx / frame_w) * 3), 2)
    back_wall_active = (row == 0)
    return active_slimpar, cr_col, back_wall_active

def send_dmx():
    frame_w, frame_h = get_frame_dimensions()

    red    = int(beam_color[2] * beam_brightness)
    green  = int(beam_color[1] * beam_brightness)
    blue   = int(beam_color[0] * beam_brightness)
    dimmer = int(beam_brightness * 255)

    pan  = int((beam_x / frame_w) * 255) if beam_x is not None else 127
    tilt = int((beam_y / frame_h) * 255) if beam_y is not None else 127

    dmx = [0] * 512

    active_slimpar, active_cr, back_wall_active = get_active_zones(
        beam_x, beam_y, frame_w, frame_h
    )

    # SlimPARs 1-8: active zone only
    if active_slimpar is not None and active_slimpar <= 8:
        base = slimpar_base(active_slimpar)
        dmx[base + 0] = red
        dmx[base + 1] = green
        dmx[base + 2] = blue

    # SlimPARs 9 & 10: back wall
    if back_wall_active:
        base9 = slimpar_base(9)
        dmx[base9 + 0] = red
        dmx[base9 + 1] = green
        dmx[base9 + 2] = blue
        base10 = slimpar_base(10)
        dmx[base10 + 0] = red
        dmx[base10 + 1] = green
        dmx[base10 + 2] = blue

    # ColorRails: active column only
    if active_cr is not None:
        cr_base = COLORRAIL_BASES[active_cr]
        dmx[cr_base] = dimmer
        for i in range(8):
            seg = cr_base + 1 + i * 3
            dmx[seg + 0] = red
            dmx[seg + 1] = green
            dmx[seg + 2] = blue
        dmx[cr_base + 25] = 0

    # Moving Head 14 (ch 109-119)
    dmx[108] = pan
    dmx[109] = tilt
    dmx[110] = 0  # Colour   
    dmx[111] = 0  
    dmx[112] = 0
    dmx[113] = 0
    dmx[114] = 11
    dmx[115] = dimmer
    dmx[116] = 128
    dmx[117] = 0
    dmx[118] = 0
    dmx[119] = 0
    dmx[120] = 0
    dmx[121] = 255
    dmx[122] = 0


    # Moving Head 15 (ch 120-130)
    dmx[123] = pan
    dmx[124] = tilt
    dmx[125] = 0   # Colour
    dmx[126] = 0
    dmx[128] = 0
    dmx[129] = 0
    dmx[130] = 11
    dmx[131] = dimmer
    dmx[132] = 128
    dmx[133] = 0
    dmx[134] = 0
    dmx[135] = 0
    dmx[136] = 255
    dmx[137] = 0

    sacn_sender[1].dmx_data = tuple(dmx)

    # Write live DMX state to JSON file
    update_dmx_json(dmx)

# =============================
# BACKGROUND DMX THREAD
# =============================

def dmx_loop():
    while True:
        try:
            send_dmx()
        except Exception as e:
            print(f"[DMX ERROR] {e}")
        threading.Event().wait(0.033)

dmx_thread = threading.Thread(target=dmx_loop, daemon=True)
dmx_thread.start()

# =============================
# CAMERA STREAMS
# =============================

def generate_raw():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_click_visual():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape
        black = np.zeros((h, w, 3), dtype=np.uint8)

        now = datetime.now()
        with lock:
            click_data[:] = [c for c in click_data if (now - c["timestamp"]).total_seconds() <= CLICK_DURATION]
            clicks = click_data.copy()

        for c in clicks:
            x = int(c["x_norm"] * w)
            y = int(c["y_norm"] * h)
            elapsed = (now - c["timestamp"]).total_seconds()
            fade = max(0, 1 - elapsed / CLICK_DURATION)
            r = 255
            g = int((1 - fade) * 255)
            cv2.circle(black, (x, y), 20, (0, g, r), -1)

        ret, buffer = cv2.imencode(".jpg", black)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_heatmap():
    global beam_x, beam_y

    while True:
        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape
        if beam_x is None:
            beam_x = w // 2
            beam_y = h // 2

        heat = np.zeros((h, w), dtype=np.float32)
        now = datetime.now()

        with lock:
            click_data[:] = [c for c in click_data if (now - c["timestamp"]).total_seconds() <= CLICK_DURATION]
            clicks = click_data.copy()

        sumx = sumy = total = 0
        for c in clicks:
            x = int(c["x_norm"] * w)
            y = int(c["y_norm"] * h)
            elapsed = (now - c["timestamp"]).total_seconds()
            weight = max(0, 1 - elapsed / CLICK_DURATION)
            sumx += x * weight
            sumy += y * weight
            total += weight
            cv2.circle(heat, (x, y), HEAT_RADIUS, weight, -1)

        if len(clicks) > 0:
            heat = cv2.GaussianBlur(heat, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

        if total > 0:
            targetx = int(sumx / total)
            targety = int(sumy / total)
            beam_x = int(beam_x * 0.85 + targetx * 0.15)
            beam_y = int(beam_y * 0.85 + targety * 0.15)

        if heat.max() > 0:
            heat = heat / heat.max()

        heat_col = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(frame, 0.7, heat_col, 0.6, 0)

        # Draw zone grid overlay
        grid_color = (200, 200, 200)
        for col in range(1, 4):
            x_line = int(col * w / 4)
            cv2.line(blend, (x_line, 0), (x_line, h), grid_color, 1)
        cv2.line(blend, (0, h // 2), (w, h // 2), grid_color, 1)
        for col in range(1, 3):
            x_line = int(col * w / 3)
            for y_seg in range(0, h, 20):
                cv2.line(blend, (x_line, y_seg), (x_line, min(y_seg + 10, h)), (100, 255, 100), 1)

        # Highlight active SlimPAR zone
        if beam_x is not None:
            active_col = min(int((beam_x / w) * 4), 3)
            active_row = 0 if beam_y < h / 2 else 1
            zone_x1 = int(active_col * w / 4)
            zone_x2 = int((active_col + 1) * w / 4)
            zone_y1 = 0 if active_row == 0 else h // 2
            zone_y2 = h // 2 if active_row == 0 else h
            cv2.rectangle(blend, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
            active_fixture = SLIMPAR_GRID[active_row][active_col]
            cv2.putText(blend, f"PAR {active_fixture}",
                        (zone_x1 + 5, zone_y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        b = int(beam_color[0] * beam_brightness)
        g = int(beam_color[1] * beam_brightness)
        r = int(beam_color[2] * beam_brightness)
        cv2.circle(blend, (beam_x, beam_y), 30, (b, g, r), -1)

        ret, buffer = cv2.imencode(".jpg", blend)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# =============================
# WEBSITE
# =============================

@app.route("/")
def index():
    return render_template_string("""
<html>
<head>
<style>
body { background:black; color:white; font-family:Arial; text-align:center; }
.panel { border:3px solid white; padding:10px; margin:20px auto; width:820px; background:#111; position:relative; }
.panel img { width:800px; cursor:pointer; }

#colourTrack{
  position:absolute; top:50%; right:40px; transform:translateY(-50%);
  width:18px; height:260px; border-radius:10px;
  background:linear-gradient(to bottom, red, yellow, lime, cyan, blue, magenta);
}
#colourDot{
  position:absolute; width:26px; height:26px; border-radius:50%;
  background:yellow; border:3px solid white; left:-4px; cursor:pointer;
}
#brightTrack{
  position:absolute; bottom:35px; left:50%; transform:translateX(-50%);
  width:300px; height:18px; border-radius:10px;
  background:linear-gradient(to right, black, white);
}
#brightDot{
  position:absolute; width:26px; height:26px; border-radius:50%;
  background:white; border:3px solid black; top:-4px; cursor:pointer;
}
</style>
</head>
<body>

<h1>Interactive Lighting System</h1>

<div class="panel">
  <h2>Live Camera</h2>
  <img id="cam" src="/video/raw">
  <div id="colourTrack"><div id="colourDot"></div></div>
  <div id="brightTrack"><div id="brightDot"></div></div>
</div>

<div class="panel">
  <h2>Click Map</h2>
  <img id="clicks" src="/video/clicks">
</div>

<div class="panel">
  <h2>Heatmap + Virtual Light</h2>
  <img id="heat" src="/video/heatmap">
</div>

<script>
function sendClick(event){
    let rect=event.target.getBoundingClientRect()
    let x=(event.clientX-rect.left)/rect.width
    let y=(event.clientY-rect.top)/rect.height
    fetch("/click",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({x_norm:x,y_norm:y})})
}
cam.onclick=sendClick
clicks.onclick=sendClick
heat.onclick=sendClick

const colourTrack=document.getElementById("colourTrack")
const colourDot=document.getElementById("colourDot")
let draggingColour=false
colourDot.onmousedown=()=>draggingColour=true
document.onmouseup=()=>{draggingColour=false;draggingBright=false}

const brightTrack=document.getElementById("brightTrack")
const brightDot=document.getElementById("brightDot")
let draggingBright=false
brightDot.onmousedown=()=>draggingBright=true

document.onmousemove=(e)=>{
    if(draggingColour){
        let rect=colourTrack.getBoundingClientRect()
        let y=e.clientY-rect.top
        y=Math.max(0,Math.min(rect.height,y))
        colourDot.style.top=(y-13)+"px"
        let hue=(y/rect.height)*360
        fetch("/set_colour",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({hue:hue})})
        colourDot.style.background="hsl("+hue+",100%,50%)"
    }
    if(draggingBright){
        let rect=brightTrack.getBoundingClientRect()
        let x=e.clientX-rect.left
        x=Math.max(0,Math.min(rect.width,x))
        brightDot.style.left=(x-13)+"px"
        let brightness=x/rect.width
        fetch("/set_brightness",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({brightness:brightness})})
    }
}
</script>
</body>
</html>
""")

# =============================
# ROUTES
# =============================

@app.route("/video/raw")
def raw():
    return Response(generate_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video/clicks")
def clicks():
    return Response(generate_click_visual(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video/heatmap")
def heatmap():
    return Response(generate_heatmap(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/click", methods=["POST"])
def click():
    global click_counter
    data = request.get_json()
    x_norm = float(data["x_norm"])
    y_norm = float(data["y_norm"])
    timestamp = datetime.now()
    with lock:
        click_counter += 1
        click_data.append({"id": click_counter, "x_norm": x_norm, "y_norm": y_norm, "timestamp": timestamp})
    print(f"[CLICK {click_counter}] {timestamp.strftime('%H:%M:%S')}")
    return jsonify({"status": "ok"})

@app.route("/set_colour", methods=["POST"])
def set_colour():
    global beam_color
    hue = float(request.get_json()["hue"])
    r, g, b = colorsys.hsv_to_rgb(hue / 360, 1, 1)
    beam_color = [int(b * 255), int(g * 255), int(r * 255)]
    return jsonify({"status": "ok"})

@app.route("/set_brightness", methods=["POST"])
def set_brightness():
    global beam_brightness
    beam_brightness = float(request.get_json()["brightness"])
    return jsonify({"status": "ok"})

# =============================
# START SERVER
# =============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)