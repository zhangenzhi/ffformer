#!/usr/bin/env python3
"""Visualize ForestFormer3D PLY prediction files as interactive HTML.

Three.js GPU-accelerated rendering with dual-panel GT vs Prediction
comparison and synchronized camera controls.

Usage:
    python tools/visualize_ply.py scene.ply
    python tools/visualize_ply.py scene.ply --mode semantic
    python tools/visualize_ply.py dir/ --batch --max-points 300000
"""
import argparse
import base64
import os
import random
import struct
import sys

SEMANTIC_COLORS = {
    0: (139, 119, 101),  # ground
    1: (210, 150, 60),   # wood
    2: (34, 180, 34),    # leaf
}
SEMANTIC_NAMES = {0: 'ground', 1: 'wood', 2: 'leaf'}

INSTANCE_PALETTE = [
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
]
UNASSIGNED_COLOR = (80, 80, 80)


def parse_ply_ascii(filepath, max_points=0):
    fields, num_vertices = [], 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break
        indices = None if (max_points <= 0 or num_vertices <= max_points) else set(
            random.sample(range(num_vertices), max_points))
        rows = []
        for i, line in enumerate(f):
            if i >= num_vertices:
                break
            if indices is not None and i not in indices:
                continue
            rows.append(line.strip().split())
    return fields, rows, num_vertices


def get_color(row, col, mode):
    if mode == 'semantic_pred':
        return SEMANTIC_COLORS.get(int(row[col['semantic_pred']]), UNASSIGNED_COLOR)
    elif mode == 'semantic_gt':
        return SEMANTIC_COLORS.get(int(row[col['semantic_gt']]), UNASSIGNED_COLOR)
    elif mode == 'instance_pred':
        v = int(row[col['instance_pred']])
        return INSTANCE_PALETTE[v % len(INSTANCE_PALETTE)] if v >= 0 else UNASSIGNED_COLOR
    elif mode == 'instance_gt':
        v = int(row[col['instance_gt']])
        return INSTANCE_PALETTE[v % len(INSTANCE_PALETTE)] if v >= 0 else UNASSIGNED_COLOR
    return UNASSIGNED_COLOR


def build_binary_data(fields, rows, left_mode, right_mode):
    """Pack: pos(3xf32) + colorL(3xU8) + colorR(3xU8) + flagL(U8) + flagR(U8) = 20 bytes."""
    col = {name: idx for idx, name in enumerate(fields)}
    n = len(rows)
    buf = bytearray(n * 20)
    for i, row in enumerate(rows):
        x = float(row[col['x']])
        y = float(row[col['y']])
        z = float(row[col['z']])
        cl = get_color(row, col, left_mode)
        cr = get_color(row, col, right_mode)
        fl = 0 if cl == UNASSIGNED_COLOR else 255
        fr = 0 if cr == UNASSIGNED_COLOR else 255
        struct.pack_into('<fff8B', buf, i * 20,
                         x, y, z,
                         cl[0], cl[1], cl[2],
                         cr[0], cr[1], cr[2],
                         fl, fr)
    return n, base64.b64encode(buf).decode('ascii')


def build_html(fields, rows, total_points, left_mode, right_mode, title):
    n, b64data = build_binary_data(fields, rows, left_mode, right_mode)
    left_label = left_mode.replace('_', ' ').title()
    right_label = right_mode.replace('_', ' ').title()

    sem_legend = ''
    for cls_id, name in SEMANTIC_NAMES.items():
        c = SEMANTIC_COLORS[cls_id]
        sem_legend += (
            f'<span class="li"><span class="ld" '
            f'style="background:rgb({c[0]},{c[1]},{c[2]})"></span>'
            f'{name}</span>')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ background:#0d1117; font-family:'Segoe UI',Arial,sans-serif; overflow:hidden; height:100vh; color:#c9d1d9 }}
#hdr {{ height:30px; display:flex; align-items:center; justify-content:center; gap:16px;
  background:linear-gradient(90deg,#161b22,#0d1117,#161b22);
  border-bottom:1px solid #30363d; font-size:13px }}
#hdr b {{ color:#e6edf3; letter-spacing:0.3px }}
#hdr span {{ color:#8b949e; font-size:11px }}
#wrap {{ display:flex; height:calc(100vh - 30px); gap:2px; background:#30363d }}
.pnl {{ flex:1; position:relative; overflow:hidden }}
.pnl canvas {{ display:block; width:100% !important; height:100% !important }}
.lbl {{ position:absolute; left:8px; top:8px; background:rgba(0,0,0,0.6);
  backdrop-filter:blur(4px); color:#fff; font-size:10px; font-weight:700;
  padding:3px 10px; border-radius:4px; pointer-events:none; z-index:5;
  letter-spacing:0.8px; text-transform:uppercase;
  border:1px solid rgba(255,255,255,0.08) }}
#cp {{ position:absolute; right:10px; top:8px; z-index:20;
  background:rgba(22,27,34,0.92); backdrop-filter:blur(6px);
  border:1px solid #30363d; border-radius:8px;
  padding:10px 12px; font-size:11px; width:150px }}
#cp label {{ display:block; margin:6px 0 2px; color:#8b949e; font-size:9px;
  text-transform:uppercase; letter-spacing:0.6px }}
#cp input[type=range] {{ width:100%; accent-color:#58a6ff }}
#cp select {{ width:100%; background:#0d1117; color:#c9d1d9;
  border:1px solid #30363d; border-radius:4px; padding:2px 4px; font-size:11px }}
#cp button {{ width:100%; margin-top:8px; padding:4px; background:#238636;
  color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:11px;
  font-weight:600 }}
#cp button:hover {{ background:#2ea043 }}
#lg {{ position:absolute; bottom:8px; left:50%; transform:translateX(-50%);
  background:rgba(22,27,34,0.88); backdrop-filter:blur(4px);
  border:1px solid #30363d;
  padding:4px 14px; border-radius:5px; z-index:10; font-size:11px; color:#c9d1d9;
  white-space:nowrap; display:flex; gap:12px }}
#err {{ display:none; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  background:#fee; border:2px solid #c00; padding:20px; border-radius:8px;
  font-size:14px; color:#c00; z-index:100; text-align:center }}
.li {{ display:inline-flex; align-items:center; gap:4px }}
.ld {{ width:9px; height:9px; border-radius:50%; display:inline-block;
  border:1px solid rgba(255,255,255,0.15) }}
</style>
<!-- Three.js r128 -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
  onerror="document.getElementById('err').style.display='block';
  document.getElementById('err').innerHTML='Failed to load Three.js from CDN. Check internet.'"></script>
</head>
<body>
<div id="hdr">
  <b>{title}</b>
  <span>{n:,} / {total_points:,} points</span>
</div>
<div id="wrap">
  <div class="pnl" id="p0"><canvas id="c0"></canvas>
    <div class="lbl">{left_label}</div></div>
  <div class="pnl" id="p1"><canvas id="c1"></canvas>
    <div class="lbl">{right_label}</div></div>
</div>
<div id="cp">
  <label>Point Size</label>
  <input type="range" id="psize" min="3" max="80" value="30">
  <label>Background</label>
  <select id="bgsel">
    <option value="#161b22">Dark</option>
    <option value="#ffffff">White</option>
    <option value="#f0efe8">Cream</option>
    <option value="#000000">Black</option>
  </select>
  <label>Unassigned Points</label>
  <select id="unsel">
    <option value="dim">Dim</option>
    <option value="hide">Hide</option>
    <option value="show">Show</option>
  </select>
  <button id="resetBtn">Reset Camera</button>
</div>
<div id="lg">
  {sem_legend}
  <span class="li"><span class="ld" style="background:#505050"></span>none</span>
</div>
<div id="err"></div>

<script>
// Wait for Three.js to load
if (typeof THREE === 'undefined') {{
  document.getElementById('err').style.display = 'block';
  document.getElementById('err').innerHTML = 'Three.js not loaded. Check internet.';
}} else {{

var N = {n};
var B64 = "{b64data}";

// Decode binary (20 bytes per point)
var raw = atob(B64);
var bytes = new Uint8Array(raw.length);
for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
var dv = new DataView(bytes.buffer);

var positions = new Float32Array(N * 3);
var colorsL = new Float32Array(N * 3);
var colorsR = new Float32Array(N * 3);
var flagsL = new Float32Array(N);
var flagsR = new Float32Array(N);

for (var i = 0; i < N; i++) {{
    var o = i * 20;
    positions[i*3]   = dv.getFloat32(o,   true);
    positions[i*3+1] = dv.getFloat32(o+4, true);
    positions[i*3+2] = dv.getFloat32(o+8, true);
    colorsL[i*3]   = bytes[o+12]/255; colorsL[i*3+1] = bytes[o+13]/255; colorsL[i*3+2] = bytes[o+14]/255;
    colorsR[i*3]   = bytes[o+15]/255; colorsR[i*3+1] = bytes[o+16]/255; colorsR[i*3+2] = bytes[o+17]/255;
    flagsL[i] = bytes[o+18] / 255.0;
    flagsR[i] = bytes[o+19] / 255.0;
}}

// Custom shader: circular points with per-point alpha + subtle depth shading
var vertShader = [
    'attribute float flag;',
    'uniform float uSize;',
    'uniform float uDimAlpha;',
    'varying vec3 vColor;',
    'varying float vAlpha;',
    'void main() {{',
    '  vColor = color;',
    '  vAlpha = flag > 0.5 ? 1.0 : uDimAlpha;',
    '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
    '  gl_PointSize = uSize * (350.0 / (-mv.z));',
    '  gl_PointSize = max(gl_PointSize, 1.0);',
    '  gl_Position = projectionMatrix * mv;',
    '}}'
].join('\\n');

var fragShader = [
    'varying vec3 vColor;',
    'varying float vAlpha;',
    'void main() {{',
    '  vec2 p = gl_PointCoord - 0.5;',
    '  float d = dot(p, p);',
    '  if (d > 0.25) discard;',
    '  float edge = smoothstep(0.25, 0.15, d);',
    '  float shade = 1.0 - d * 0.5;',
    '  gl_FragColor = vec4(vColor * shade, vAlpha * edge);',
    '}}'
].join('\\n');

// Center point cloud
var cx = 0, cy = 0, cz = 0;
for (var i = 0; i < N; i++) {{ cx += positions[i*3]; cy += positions[i*3+1]; cz += positions[i*3+2]; }}
cx /= N; cy /= N; cz /= N;
for (var i = 0; i < N; i++) {{ positions[i*3] -= cx; positions[i*3+1] -= cy; positions[i*3+2] -= cz; }}

var R = 0;
for (var i = 0; i < N; i++) {{
    var dx = positions[i*3], dy = positions[i*3+1], dz = positions[i*3+2];
    R = Math.max(R, Math.sqrt(dx*dx + dy*dy + dz*dz));
}}

var scenes = [], cameras = [], renderers = [], pointMats = [];
var canvases = [document.getElementById('c0'), document.getElementById('c1')];
var panels = [document.getElementById('p0'), document.getElementById('p1')];
var colorData = [colorsL, colorsR];
var flagData = [flagsL, flagsR];
var curBg = 0x161b22;

// Camera orbit state (shared between both views)
var spherical = {{ theta: -Math.PI/2, phi: Math.PI/3, radius: R * 2.8 }};
var target = new THREE.Vector3(0, 0, 0);

function updateCameras() {{
    var sp = Math.sin(spherical.phi);
    var cp = Math.cos(spherical.phi);
    var st = Math.sin(spherical.theta);
    var ct = Math.cos(spherical.theta);
    var off = new THREE.Vector3(
        spherical.radius * sp * ct,
        spherical.radius * sp * st,
        spherical.radius * cp
    );
    for (var i = 0; i < 2; i++) {{
        cameras[i].position.copy(target).add(off);
        cameras[i].lookAt(target);
    }}
}}

for (var i = 0; i < 2; i++) {{
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(curBg);
    scenes.push(scene);

    var w = panels[i].clientWidth;
    var h = panels[i].clientHeight;

    var camera = new THREE.PerspectiveCamera(60, w / h, 0.01, R * 50);
    camera.up.set(0, 0, 1);
    cameras.push(camera);

    var renderer = new THREE.WebGLRenderer({{ canvas: canvases[i], antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h, false);
    renderers.push(renderer);

    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions.slice(), 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colorData[i], 3));
    geo.setAttribute('flag', new THREE.Float32BufferAttribute(flagData[i], 1));

    var mat = new THREE.ShaderMaterial({{
        uniforms: {{
            uSize: {{ value: R * 0.010 }},
            uDimAlpha: {{ value: 0.12 }}
        }},
        vertexShader: vertShader,
        fragmentShader: fragShader,
        vertexColors: true,
        transparent: true,
        depthWrite: true,
        depthTest: true
    }});
    pointMats.push(mat);
    scene.add(new THREE.Points(geo, mat));
}}

updateCameras();

// Save initial state for reset
var initSpherical = {{ theta: spherical.theta, phi: spherical.phi, radius: spherical.radius }};
var initTarget = target.clone();

// --- Mouse controls on BOTH panels ---
var isDragging = false, isPanning = false;
var lastX = 0, lastY = 0;

function onMouseDown(e) {{
    if (e.button === 0) isDragging = true;      // left = rotate
    if (e.button === 2) isPanning = true;        // right = pan
    lastX = e.clientX;
    lastY = e.clientY;
    e.preventDefault();
}}
function onMouseMove(e) {{
    var dx = e.clientX - lastX;
    var dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;

    if (isDragging) {{
        spherical.theta -= dx * 0.005;
        spherical.phi -= dy * 0.005;
        spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi));
        updateCameras();
    }}
    if (isPanning) {{
        var cam = cameras[0];
        var right = new THREE.Vector3();
        var up = new THREE.Vector3();
        cam.getWorldDirection(new THREE.Vector3());
        right.crossVectors(cam.getWorldDirection(new THREE.Vector3()), cam.up).normalize();
        up.copy(cam.up).normalize();
        var panSpeed = spherical.radius * 0.002;
        target.add(right.multiplyScalar(-dx * panSpeed));
        target.add(up.multiplyScalar(dy * panSpeed));
        updateCameras();
    }}
}}
function onMouseUp(e) {{
    isDragging = false;
    isPanning = false;
}}
function onWheel(e) {{
    spherical.radius *= (1 + e.deltaY * 0.001);
    spherical.radius = Math.max(R * 0.1, Math.min(R * 20, spherical.radius));
    updateCameras();
    e.preventDefault();
}}
function onContext(e) {{ e.preventDefault(); }}

for (var i = 0; i < 2; i++) {{
    panels[i].addEventListener('mousedown', onMouseDown);
    panels[i].addEventListener('contextmenu', onContext);
    panels[i].addEventListener('wheel', onWheel, {{ passive: false }});
}}
document.addEventListener('mousemove', onMouseMove);
document.addEventListener('mouseup', onMouseUp);

function onResize() {{
    for (var i = 0; i < 2; i++) {{
        var w = panels[i].clientWidth;
        var h = panels[i].clientHeight;
        if (w < 1 || h < 1) continue;
        cameras[i].aspect = w / h;
        cameras[i].updateProjectionMatrix();
        renderers[i].setSize(w, h, false);
    }}
}}
window.addEventListener('resize', onResize);
onResize();

// UI Controls
document.getElementById('psize').oninput = function() {{
    var v = this.value / 30.0;
    for (var i = 0; i < 2; i++) pointMats[i].uniforms.uSize.value = R * 0.010 * v;
}};
document.getElementById('bgsel').onchange = function() {{
    var c = new THREE.Color(this.value);
    curBg = c;
    for (var i = 0; i < 2; i++) scenes[i].background = c;
}};
document.getElementById('unsel').onchange = function() {{
    var mode = this.value;
    var alpha = mode === 'hide' ? 0.0 : mode === 'dim' ? 0.12 : 1.0;
    for (var i = 0; i < 2; i++) pointMats[i].uniforms.uDimAlpha.value = alpha;
}};
document.getElementById('resetBtn').onclick = function() {{
    spherical.theta = initSpherical.theta;
    spherical.phi = initSpherical.phi;
    spherical.radius = initSpherical.radius;
    target.copy(initTarget);
    updateCameras();
}};

// Render loop
function animate() {{
    requestAnimationFrame(animate);
    for (var i = 0; i < 2; i++) {{
        renderers[i].render(scenes[i], cameras[i]);
    }}
}}
animate();

}} // end if THREE
</script>
</body>
</html>"""
    return html


def process_file(ply_path, output_dir, max_points, mode):
    basename = os.path.splitext(os.path.basename(ply_path))[0]
    print(f'Processing: {basename} ...', end=' ', flush=True)
    fields, rows, total = parse_ply_ascii(ply_path, max_points)
    print(f'{len(rows):,}/{total:,} points loaded.', end=' ', flush=True)

    if mode == 'instance':
        left_mode, right_mode = 'instance_gt', 'instance_pred'
    elif mode == 'semantic':
        left_mode, right_mode = 'semantic_gt', 'semantic_pred'
    elif mode == 'compare':
        left_mode, right_mode = 'instance_gt', 'semantic_pred'
    else:
        left_mode, right_mode = mode, mode

    html = build_html(fields, rows, total, left_mode, right_mode, basename)
    out_path = os.path.join(output_dir, f'{basename}_{mode}.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'-> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ForestFormer3D PLY predictions (Three.js)')
    parser.add_argument('input', help='PLY file or directory')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--max-points', type=int, default=0,
                        help='Max points to render (0 = all)')
    parser.add_argument('--mode', default='instance',
                        choices=['instance', 'semantic', 'compare',
                                 'instance_pred', 'instance_gt',
                                 'semantic_pred', 'semantic_gt'])
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        ply_dir = args.input
        ply_files = sorted(os.path.join(ply_dir, f)
                           for f in os.listdir(ply_dir) if f.endswith('.ply'))
        if not ply_files:
            print(f'No .ply files found in {ply_dir}')
            sys.exit(1)
        print(f'Found {len(ply_files)} PLY files.')
    else:
        ply_files = [args.input]

    output_dir = args.output_dir or (
        args.input if os.path.isdir(args.input)
        else os.path.dirname(args.input) or '.')
    os.makedirs(output_dir, exist_ok=True)

    for p in ply_files:
        process_file(p, output_dir, args.max_points, args.mode)
    print('Done!')


if __name__ == '__main__':
    main()
