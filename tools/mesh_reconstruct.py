#!/usr/bin/env python3
"""Render ForestFormer3D predictions as shaded solid-looking point clouds.

Uses torch GPU for fast KNN-based normal estimation (handles 100M+ points),
with chunked processing to avoid OOM. Falls back to Open3D CPU if no GPU.

Usage:
    python tools/mesh_reconstruct.py work_dirs/test_single/scene.ply
    python tools/mesh_reconstruct.py scene.ply --max-points 5000000
    python tools/mesh_reconstruct.py scene.ply --device cpu
"""
import argparse
import base64
import os
import struct
import sys
import time

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

INSTANCE_COLORS = np.array([
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
], dtype=np.uint8)
SEMANTIC_COLORS = {0: (110,95,80), 1: (180,130,50), 2: (40,160,40)}
UNASSIGNED_COLOR = np.array([60, 60, 60], dtype=np.uint8)


def parse_ply_numpy(filepath):
    """Fast numpy-based ASCII PLY parser."""
    fields = []
    num_verts = 0
    header_lines = 0

    with open(filepath, 'r') as f:
        for line in f:
            header_lines += 1
            line = line.strip()
            if line.startswith('element vertex'):
                num_verts = int(line.split()[-1])
            elif line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break

    print(f'  {num_verts:,} points, parsing ...', end=' ', flush=True)
    t0 = time.time()
    data = np.loadtxt(filepath, skiprows=header_lines, max_rows=num_verts)
    print(f'{time.time() - t0:.1f}s')

    col = {name: idx for idx, name in enumerate(fields)}
    return data, col, num_verts


def voxel_downsample(xyz, max_points):
    """Voxel-based downsampling preserving spatial structure."""
    if max_points <= 0 or len(xyz) <= max_points:
        return np.arange(len(xyz))

    lo, hi = 0.05, 50.0
    best_idx = None
    for _ in range(20):
        mid = (lo + hi) / 2
        voxel_ids = np.floor(xyz / mid).astype(np.int64)
        packed = np.ascontiguousarray(voxel_ids).view(
            np.dtype((np.void, voxel_ids.dtype.itemsize * 3)))
        _, idx = np.unique(packed, return_index=True)
        if len(idx) > max_points:
            lo = mid
        else:
            hi = mid
            best_idx = idx
    if best_idx is None or len(best_idx) > max_points:
        best_idx = np.random.choice(len(xyz), max_points, replace=False)
    return np.sort(best_idx)


def estimate_normals_torch(xyz, k=30, chunk_size=100_000, device='cuda'):
    """GPU-accelerated KNN normal estimation using torch.

    Processes in chunks to handle arbitrarily large point clouds.
    """
    N = len(xyz)
    normals = np.zeros_like(xyz)
    xyz_t = torch.from_numpy(xyz).float().to(device)

    print(f'  Estimating normals on {device} ({N:,} pts, k={k}) ...', flush=True)
    t0 = time.time()

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = xyz_t[start:end]  # (C, 3)

        # KNN via batched cdist (chunked to avoid OOM on distance matrix)
        # For each query point, find k nearest neighbors
        knn_chunk = min(chunk_size * 4, N)  # search window
        best_normals = torch.zeros(end - start, 3, device=device)

        # Use a sliding search window around the chunk
        search_start = max(0, start - knn_chunk // 2)
        search_end = min(N, end + knn_chunk // 2)
        if search_end - search_start < k:
            search_start, search_end = 0, N
        search_pts = xyz_t[search_start:search_end]

        # Compute distances in sub-chunks to limit memory
        sub_chunk = 10_000
        for ss in range(0, end - start, sub_chunk):
            se = min(ss + sub_chunk, end - start)
            q = chunk[ss:se]  # (sc, 3)

            # Pairwise distances
            dists = torch.cdist(q, search_pts)  # (sc, search_N)
            _, knn_idx = dists.topk(k, largest=False)  # (sc, k)

            # Gather neighbor points
            neighbors = search_pts[knn_idx]  # (sc, k, 3)

            # PCA for normals: covariance of neighbors
            centered = neighbors - neighbors.mean(dim=1, keepdim=True)
            cov = torch.bmm(centered.transpose(1, 2), centered) / k  # (sc, 3, 3)

            # Eigendecomposition — normal is eigenvector of smallest eigenvalue
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            best_normals[ss:se] = eigenvectors[:, :, 0]  # smallest eigenvalue

        normals[start:end] = best_normals.cpu().numpy()

        pct = end * 100 // N
        elapsed = time.time() - t0
        if end < N:
            eta = elapsed / end * (N - end)
            print(f'\r  Normals: {pct}% ({end:,}/{N:,}) ETA {eta:.0f}s  ',
                  end='', flush=True)

    # Orient normals: flip those pointing downward
    down_mask = normals[:, 2] < -0.5
    normals[down_mask] *= -1

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals /= norms

    print(f'\r  Normals done: {time.time() - t0:.1f}s                    ')
    return normals.astype(np.float32)


def estimate_normals_cpu(xyz, radius=0.5, max_nn=30):
    """Fallback: Open3D CPU normal estimation."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    down_mask = normals[:, 2] < -0.5
    normals[down_mask] *= -1
    return normals


def main():
    parser = argparse.ArgumentParser(
        description='Render tree point clouds with normal-based shading')
    parser.add_argument('input', help='PLY prediction file')
    parser.add_argument('--max-points', type=int, default=5_000_000,
                        help='Max points for HTML viewer (default: 5M, 0 = all)')
    parser.add_argument('--device', default='auto',
                        help='Device for normal estimation: cuda, cpu, or auto')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
    else:
        device = args.device
    print(f'Device: {device}')

    print(f'Loading {args.input} ...')
    data, col, n_total = parse_ply_numpy(args.input)
    xyz = data[:, [col['x'], col['y'], col['z']]].astype(np.float32)
    sem = data[:, col['semantic_pred']].astype(int)
    inst = data[:, col['instance_pred']].astype(int)

    # Subsample if needed
    if args.max_points > 0 and n_total > args.max_points:
        print(f'  Voxel downsampling {n_total:,} -> {args.max_points:,} ...',
              end=' ', flush=True)
        t0 = time.time()
        idx = voxel_downsample(xyz, args.max_points)
        xyz = xyz[idx]; sem = sem[idx]; inst = inst[idx]
        print(f'{len(xyz):,} points, {time.time() - t0:.1f}s')

    N = len(xyz)

    # Assign colors vectorized
    colors = np.tile(UNASSIGNED_COLOR, (N, 1))
    tree_mask = inst >= 0
    colors[tree_mask] = INSTANCE_COLORS[inst[tree_mask] % len(INSTANCE_COLORS)]
    ground_mask = (~tree_mask) & (sem == 0)
    colors[ground_mask] = np.array(SEMANTIC_COLORS[0], dtype=np.uint8)

    # Estimate normals
    if device == 'cuda' and HAS_TORCH:
        normals = estimate_normals_torch(xyz, k=30, device=device)
    else:
        print('  Using Open3D CPU for normals (slower) ...')
        normals = estimate_normals_cpu(xyz)

    # Pack binary: pos(3xf32) + normal(3xf16) + color(3xU8) + flag(U8)
    # = 12 + 6 + 3 + 1 = 22 bytes per point
    print('Building viewer ...', end=' ', flush=True)
    t0 = time.time()

    # Vectorized packing
    pos_bytes = xyz.tobytes()
    normal_f16 = normals.astype(np.float16)
    norm_bytes = normal_f16.tobytes()

    flag_arr = np.zeros(N, dtype=np.uint8)
    flag_arr[tree_mask] = 255
    flag_arr[ground_mask] = 80

    buf = bytearray(N * 22)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(N, 22)
    arr[:, 0:12] = np.frombuffer(pos_bytes, dtype=np.uint8).reshape(N, 12)
    arr[:, 12:18] = np.frombuffer(norm_bytes, dtype=np.uint8).reshape(N, 6)
    arr[:, 18:21] = colors
    arr[:, 21] = flag_arr

    b64 = base64.b64encode(bytes(buf)).decode('ascii')
    print(f'{time.time() - t0:.1f}s')

    n_trees = len(set(inst[inst >= 0]))
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or os.path.dirname(args.input) or '.'
    os.makedirs(output_dir, exist_ok=True)

    js = r"""
var N = __N__;
var raw = Uint8Array.from(atob("__B64__"), function(c){ return c.charCodeAt(0); });
var dv = new DataView(raw.buffer);

var pos = new Float32Array(N*3);
var norm = new Float32Array(N*3);
var col = new Float32Array(N*3);
var flag = new Float32Array(N);

for(var i=0;i<N;i++){
    var o=i*22;
    pos[i*3]=dv.getFloat32(o,true);
    pos[i*3+1]=dv.getFloat32(o+4,true);
    pos[i*3+2]=dv.getFloat32(o+8,true);
    // float16 decode
    var hx=dv.getUint16(o+12,true), hy=dv.getUint16(o+14,true), hz=dv.getUint16(o+16,true);
    norm[i*3]=f16(hx); norm[i*3+1]=f16(hy); norm[i*3+2]=f16(hz);
    col[i*3]=raw[o+18]/255; col[i*3+1]=raw[o+19]/255; col[i*3+2]=raw[o+20]/255;
    flag[i]=raw[o+21]/255;
}

function f16(h){
    var s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x3ff;
    if(e===0) return (s?-1:1)*Math.pow(2,-14)*(m/1024);
    if(e===31) return m?NaN:(s?-Infinity:Infinity);
    return (s?-1:1)*Math.pow(2,e-15)*(1+m/1024);
}

// Center
var cx=0,cy=0,cz=0;
for(var i=0;i<N;i++){cx+=pos[i*3];cy+=pos[i*3+1];cz+=pos[i*3+2];}
cx/=N;cy/=N;cz/=N;
for(var i=0;i<N;i++){pos[i*3]-=cx;pos[i*3+1]-=cy;pos[i*3+2]-=cz;}
var R=0;
for(var i=0;i<N;i++){
    var dx=pos[i*3],dy=pos[i*3+1],dz=pos[i*3+2];
    R=Math.max(R,Math.sqrt(dx*dx+dy*dy+dz*dz));
}

var canvas=document.getElementById('c0');
var el=canvas.parentElement;
var w=el.clientWidth, h=el.clientHeight;

var scene=new THREE.Scene();
scene.background=new THREE.Color(0x161b22);

var camera=new THREE.PerspectiveCamera(55,w/h,0.01,R*50);
camera.up.set(0,0,1);

var renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(w,h,false);

// Shaders: circular splat with Phong lighting
var vtx=[
'attribute vec3 norm;',
'attribute float flag;',
'uniform float uSize;',
'uniform float uDimAlpha;',
'varying vec3 vColor;',
'varying vec3 vNormal;',
'varying float vAlpha;',
'varying vec3 vViewPos;',
'void main(){',
'  vColor=color;',
'  vNormal=normalMatrix*norm;',
'  vAlpha=flag>0.1 ? 1.0 : uDimAlpha;',
'  vec4 mv=modelViewMatrix*vec4(position,1.0);',
'  vViewPos=mv.xyz;',
'  gl_PointSize=uSize*(400.0/(-mv.z));',
'  gl_PointSize=max(gl_PointSize,1.5);',
'  gl_Position=projectionMatrix*mv;',
'}'
].join('\n');

var frg=[
'uniform vec3 uLightDir;',
'uniform vec3 uLightColor;',
'uniform vec3 uAmbient;',
'varying vec3 vColor;',
'varying vec3 vNormal;',
'varying float vAlpha;',
'varying vec3 vViewPos;',
'void main(){',
'  vec2 p=gl_PointCoord-0.5;',
'  float d2=dot(p,p);',
'  if(d2>0.25) discard;',
'  float edge=smoothstep(0.25,0.12,d2);',
// Normal-based Phong lighting
'  vec3 n=normalize(vNormal);',
'  vec3 ldir=normalize(uLightDir);',
// Diffuse
'  float diff=max(dot(n,ldir),0.0);',
// Specular
'  vec3 viewDir=normalize(-vViewPos);',
'  vec3 halfDir=normalize(ldir+viewDir);',
'  float spec=pow(max(dot(n,halfDir),0.0),32.0)*0.3;',
// Hemisphere ambient (sky blue + ground brown)
'  float hemi=0.5+0.5*n.z;',
'  vec3 hemiColor=mix(vec3(0.15,0.12,0.1),vec3(0.2,0.25,0.35),hemi);',
// Combine
'  vec3 lit=vColor*(uAmbient+hemiColor+uLightColor*diff)+uLightColor*spec;',
// Subtle depth darkening at edges of splat
'  lit*=(1.0-d2*0.3);',
'  gl_FragColor=vec4(lit,vAlpha*edge);',
'}'
].join('\n');

var geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
geo.setAttribute('norm',new THREE.Float32BufferAttribute(norm,3));
geo.setAttribute('flag',new THREE.Float32BufferAttribute(flag,1));

var mat=new THREE.ShaderMaterial({
    uniforms:{
        uSize:{value:R*0.015},
        uDimAlpha:{value:0.15},
        uLightDir:{value:new THREE.Vector3(0.5,-0.7,0.9).normalize()},
        uLightColor:{value:new THREE.Color(1.0,0.95,0.85)},
        uAmbient:{value:new THREE.Color(0.18,0.18,0.22)}
    },
    vertexShader:vtx,
    fragmentShader:frg,
    vertexColors:true,
    transparent:true,
    depthWrite:true,
    depthTest:true
});
scene.add(new THREE.Points(geo,mat));

// Camera orbit
var sph={theta:-Math.PI/2, phi:Math.PI/4, radius:R*3.0};
var tgt=new THREE.Vector3(0,0,0);
function updCam(){
    var sp=Math.sin(sph.phi),cp=Math.cos(sph.phi);
    var st=Math.sin(sph.theta),ct=Math.cos(sph.theta);
    camera.position.set(tgt.x+sph.radius*sp*ct, tgt.y+sph.radius*sp*st, tgt.z+sph.radius*cp);
    camera.lookAt(tgt);
}
updCam();
var initS={theta:sph.theta,phi:sph.phi,radius:sph.radius};
var initT=tgt.clone();

// Mouse
var isDrag=false,isPan=false,lx=0,ly=0;
el.addEventListener('mousedown',function(e){
    if(e.button===0)isDrag=true;if(e.button===2)isPan=true;
    lx=e.clientX;ly=e.clientY;e.preventDefault();
    autoRotate=false;document.getElementById('autorot').checked=false;
});
document.addEventListener('mousemove',function(e){
    var dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
    if(isDrag){sph.theta-=dx*0.005;sph.phi-=dy*0.005;
        sph.phi=Math.max(0.05,Math.min(Math.PI-0.05,sph.phi));updCam();}
    if(isPan){var right=new THREE.Vector3(),up=new THREE.Vector3();
        right.crossVectors(camera.getWorldDirection(new THREE.Vector3()),camera.up).normalize();
        up.copy(camera.up).normalize();var ps=sph.radius*0.002;
        tgt.add(right.multiplyScalar(-dx*ps));tgt.add(up.multiplyScalar(dy*ps));updCam();}
});
document.addEventListener('mouseup',function(){isDrag=false;isPan=false;});
el.addEventListener('wheel',function(e){
    sph.radius*=(1+e.deltaY*0.001);
    sph.radius=Math.max(R*0.2,Math.min(R*20,sph.radius));updCam();e.preventDefault();
},{passive:false});
el.addEventListener('contextmenu',function(e){e.preventDefault();});

function onResize(){
    var w2=el.clientWidth,h2=el.clientHeight;
    camera.aspect=w2/h2;camera.updateProjectionMatrix();
    renderer.setSize(w2,h2,false);
}
window.addEventListener('resize',onResize);
setTimeout(onResize,100);

// UI
document.getElementById('psize').oninput=function(){
    mat.uniforms.uSize.value=R*0.015*(this.value/30.0);};
document.getElementById('bgsel').onchange=function(){
    scene.background=new THREE.Color(this.value);};
document.getElementById('unsel').onchange=function(){
    var v=this.value;
    mat.uniforms.uDimAlpha.value=v==='hide'?0.0:v==='dim'?0.15:1.0;};
document.getElementById('resetBtn').onclick=function(){
    sph.theta=initS.theta;sph.phi=initS.phi;sph.radius=initS.radius;
    tgt.copy(initT);updCam();};

var autoRotate=true;
document.getElementById('autorot').onchange=function(){autoRotate=this.checked;};

(function anim(){
    requestAnimationFrame(anim);
    if(autoRotate){sph.theta+=0.003;updCam();}
    renderer.render(scene,camera);
})();
"""
    js = js.replace('__N__', str(N))
    js = js.replace('__B64__', b64)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{basename} - Shaded 3D</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;font-family:'Segoe UI',Arial,sans-serif;overflow:hidden;height:100vh;color:#c9d1d9}}
#hdr{{height:30px;display:flex;align-items:center;justify-content:center;gap:16px;
  background:linear-gradient(90deg,#161b22,#0d1117,#161b22);
  border-bottom:1px solid #30363d;font-size:13px}}
#hdr b{{color:#e6edf3}}
#hdr span{{color:#8b949e;font-size:11px}}
.wrap{{height:calc(100vh - 30px);position:relative}}
.wrap canvas{{display:block;width:100%!important;height:100%!important}}
#cp{{position:absolute;right:10px;top:10px;z-index:20;
  background:rgba(22,27,34,0.92);backdrop-filter:blur(6px);
  border:1px solid #30363d;border-radius:8px;
  padding:10px 12px;font-size:11px;width:155px}}
#cp label{{display:block;margin:6px 0 2px;color:#8b949e;font-size:9px;
  text-transform:uppercase;letter-spacing:0.6px}}
#cp input[type=range]{{width:100%;accent-color:#58a6ff}}
#cp select{{width:100%;background:#0d1117;color:#c9d1d9;
  border:1px solid #30363d;border-radius:4px;padding:2px 4px;font-size:11px}}
#cp button{{width:100%;margin-top:8px;padding:4px;background:#238636;
  color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;font-weight:600}}
#cp button:hover{{background:#2ea043}}
.cb{{display:flex;align-items:center;gap:6px;margin-top:6px}}
.cb input{{accent-color:#58a6ff}}
#info{{position:absolute;left:10px;bottom:10px;background:rgba(22,27,34,0.88);
  border:1px solid #30363d;border-radius:6px;
  padding:6px 12px;font-size:11px;color:#8b949e}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
  onerror="document.body.innerHTML='<h2 style=color:red;padding:40px>Failed to load Three.js</h2>'"></script>
</head>
<body>
<div id="hdr">
  <b>{basename}</b>
  <span>Shaded Point Cloud &mdash; {n_trees} trees, {N:,} / {n_total:,} points</span>
</div>
<div class="wrap" id="p0"><canvas id="c0"></canvas></div>
<div id="cp">
  <label>Point Size</label>
  <input type="range" id="psize" min="5" max="80" value="30">
  <label>Background</label>
  <select id="bgsel">
    <option value="#161b22">Dark</option>
    <option value="#ffffff">White</option>
    <option value="#000000">Black</option>
    <option value="#1a1a2e">Navy</option>
  </select>
  <label>Unassigned</label>
  <select id="unsel">
    <option value="dim">Dim</option>
    <option value="hide">Hide</option>
    <option value="show">Show</option>
  </select>
  <div class="cb"><input type="checkbox" id="autorot" checked>
    <label style="margin:0;display:inline">Auto Rotate</label></div>
  <button id="resetBtn">Reset Camera</button>
</div>
<div id="info">{n_trees} trees &bull; {N:,} points &bull; normal-shaded splats</div>
<script>
if(typeof THREE!=='undefined'){{
{js}
}}
</script>
</body>
</html>"""

    out_path = os.path.join(output_dir, f'{basename}_render.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'  -> {out_path}')
    print('Done!')


if __name__ == '__main__':
    main()
