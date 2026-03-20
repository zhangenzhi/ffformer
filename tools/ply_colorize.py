#!/usr/bin/env python3
"""Add RGB colors to ForestFormer3D PLY based on instance/semantic predictions.

Outputs a new PLY with red/green/blue properties that CloudCompare can display.

Usage:
    python tools/ply_colorize.py work_dirs/jpeaks_area1/jpeaks_area1_test.ply
    python tools/ply_colorize.py scene.ply --mode semantic
    python tools/ply_colorize.py scene.ply -o colored.ply
"""
import argparse
import os
import sys
import time

import numpy as np

INSTANCE_PALETTE = np.array([
    (228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),
    (255,255,51),(166,86,40),(247,129,191),(153,153,153),(102,194,165),
    (252,141,98),(141,160,203),(231,41,138),(34,139,34),(210,180,140),
    (0,191,255),(221,160,221),(127,255,0),(255,215,0),(70,130,180),
    (240,128,128),(176,224,230),(205,133,63),(188,189,34),(148,0,211),
    (0,206,209),
], dtype=np.uint8)

SEMANTIC_COLORS = np.array([
    [139, 119, 101],  # 0: ground
    [210, 150, 60],   # 1: wood
    [34, 180, 34],    # 2: leaf
], dtype=np.uint8)

UNASSIGNED = np.array([80, 80, 80], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Add RGB colors to PLY')
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('-o', '--output', help='Output PLY (default: <input>_colored.ply)')
    parser.add_argument('--mode', default='instance', choices=['instance', 'semantic'],
                        help='Color by instance or semantic labels')
    args = parser.parse_args()

    output = args.output or args.input.replace('.ply', f'_{args.mode}_colored.ply')

    # Parse header
    fields = []
    num_vertices = 0
    header_lines = []
    with open(args.input, 'r') as f:
        for line in f:
            header_lines.append(line)
            stripped = line.strip()
            if stripped.startswith('element vertex'):
                num_vertices = int(stripped.split()[-1])
            elif stripped.startswith('property'):
                fields.append(stripped.split()[-1])
            elif stripped == 'end_header':
                break

    col = {name: idx for idx, name in enumerate(fields)}
    print(f'Input:  {args.input} ({num_vertices:,} points)')
    print(f'Output: {output}')
    print(f'Mode:   {args.mode}')

    # Build new header with RGB
    new_header = []
    for line in header_lines:
        stripped = line.strip()
        if stripped == 'end_header':
            new_header.append('property uchar red\n')
            new_header.append('property uchar green\n')
            new_header.append('property uchar blue\n')
            new_header.append('end_header\n')
        else:
            new_header.append(line)

    # Stream process: read input, colorize, write output
    print('Processing ...', flush=True)
    t0 = time.time()

    with open(args.input, 'r') as fin, open(output, 'w') as fout:
        # Skip input header
        for line in fin:
            if line.strip() == 'end_header':
                break

        # Write new header
        for line in new_header:
            fout.write(line)

        # Process in chunks for speed
        buf = []
        count = 0
        chunk = 1_000_000

        for line in fin:
            if count >= num_vertices:
                break
            buf.append(line.strip())
            count += 1

            if len(buf) >= chunk:
                _write_chunk(buf, col, args.mode, fout)
                pct = count * 100 // num_vertices
                elapsed = time.time() - t0
                eta = elapsed / count * (num_vertices - count)
                print(f'\r  {pct}% ({count:,}/{num_vertices:,}) ETA {eta:.0f}s  ',
                      end='', flush=True)
                buf = []

        if buf:
            _write_chunk(buf, col, args.mode, fout)

    print(f'\r  Done: {time.time() - t0:.1f}s, {output}                    ')


def _write_chunk(lines, col, mode, fout):
    """Colorize a chunk and write to output."""
    n = len(lines)
    split_lines = [l.split() for l in lines]

    for vals in split_lines:
        if mode == 'instance' and 'instance_pred' in col:
            inst = int(vals[col['instance_pred']])
            if inst >= 0:
                c = INSTANCE_PALETTE[inst % len(INSTANCE_PALETTE)]
            else:
                # Fall back to semantic for unassigned
                if 'semantic_pred' in col:
                    sem = int(vals[col['semantic_pred']])
                    c = SEMANTIC_COLORS[sem] if 0 <= sem < len(SEMANTIC_COLORS) else UNASSIGNED
                else:
                    c = UNASSIGNED
        elif mode == 'semantic' and 'semantic_pred' in col:
            sem = int(vals[col['semantic_pred']])
            c = SEMANTIC_COLORS[sem] if 0 <= sem < len(SEMANTIC_COLORS) else UNASSIGNED
        else:
            c = UNASSIGNED

        fout.write(' '.join(vals) + f' {c[0]} {c[1]} {c[2]}\n')


if __name__ == '__main__':
    main()
