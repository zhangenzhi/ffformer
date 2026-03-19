#!/usr/bin/env python3
"""Render ForestFormer3D predictions as shaded solid-looking point clouds.

Instead of mesh reconstruction (which deforms tree shapes), this uses
normal-estimated point splatting with Phong lighting — each point becomes
a small shaded disc oriented by its surface normal, giving a solid 3D
appearance while preserving the true tree shape.

Usage:
    python tools/mesh_reconstruct.py work_dirs/test_single/scene.ply
    python tools/mesh_reconstruct.py scene.ply --max-points 300000
"""
import argparse
import base64
import os
import struct
import sys
import time

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d required.")
    sys.exit(1)


INSTANCE_COLORS = [
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
]
SEMANTIC_COLORS = {0: (110,95,80), 1: (180,130,50), 2: (40,160,40)}
UNASSIGNED_COLOR = (60, 60, 60)


def parse_ply(filepath):
    fields, num_verts = [], 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                num_verts = int(line.split()[-1])
            elif line.startswith('property'):
                fields.append(line.split()[-1])
            elif line == 'end_header':
                break
        data = {fn: [] for fn in fields}
        for i, line in enumerate(f):
            if i >= num_verts:
                break
            vals = line.strip().split()
            for j, fn in enumerate(fields):
                data[fn].append(float(vals[j]))
    for fn in fields:
        data[fn] = np.array(data[fn])
    return data, num_verts


def estimate_normals(xyz, radius=0.5, max_nn=30):
    """Estimate normals using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    # Flip normals that point downward (trees generally face outward/up)
    down_mask = normals[:, 2] < -0.5
    normals[down_mask] *= -1
    return normals


def main():
    parser = argparse.ArgumentParser(
        description='Render tree point clouds with normal-based shading')
    parser.add_argument('input', help='PLY prediction file')
    parser.add_argument('--max-points', type=int, default=0,
                        help='Max points to render (0 = all)')
    parser.add_argument('--output-dir')
    args = parser.parse_args()

    print(f'Loading {args.input} ...')
    data, n_total = parse_ply(args.input)
    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
    sem = data['semantic_pred'].astype(int)
    inst = data['instance_pred'].astype(int)
    print(f'  {n_total:,} points')

    # Subsample if needed
    if args.max_points > 0 and n_total > args.max_points:
        idx = np.random.choice(n_total, args.max_points, replace=False)
        xyz = xyz[idx]; sem = sem[idx]; inst = inst[idx]
        print(f'  Subsampled to {args.max_points:,}')

    N = len(xyz)

    # Assign colors by instance (trees) or semantic (ground)
    colors = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        if inst[i] >= 0:
            c = INSTANCE_COLORS[int(inst[i]) % len(INSTANCE_COLORS)]
        else:
            c = SEMANTIC_COLORS.get(int(sem[i]), UNASSIGNED_COLOR)
        colors[i] = c

    # Estimate normals
    print('Estimating normals ...')
    t0 = time.time()
    normals = estimate_normals(xyz)
    print(f'  Done ({time.time()-t0:.1f}s)')

    # Pack binary: pos(3xf32) + normal(3xf16) + color(3xU8) + flag(U8)
    # = 12 + 6 + 3 + 1 = 22 bytes per point
    print('Building viewer ...')
    buf = bytearray(N * 22)
    for i in range(N):
        nx = int(np.float16(normals[i, 0]).view(np.uint16))
        ny = int(np.float16(normals[i, 1]).view(np.uint16))
        nz = int(np.float16(normals[i, 2]).view(np.uint16))
        flag = 255 if inst[i] >= 0 else (80 if sem[i] == 0 else 0)
        struct.pack_into('<fff3H3BB', buf, i * 22,
                         xyz[i, 0], xyz[i, 1], xyz[i, 2],
                         nx, ny, nz,
                         colors[i, 0], colors[i, 1], colors[i, 2],
                         flag)

    b64 = base64.b64encode(buf).decode('ascii')

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
  <span>Shaded Point Cloud &mdash; {n_trees} trees, {N:,} points</span>
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
