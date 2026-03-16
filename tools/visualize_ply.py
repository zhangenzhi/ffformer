#!/usr/bin/env python3
"""Visualize ForestFormer3D PLY prediction files as interactive HTML.

Usage:
    # Visualize a single file (subsample to 100k points by default):
    python tools/visualize_ply.py work_dirs/test_output/RMIT_RMIT_test_test.ply

    # Visualize with custom max points:
    python tools/visualize_ply.py work_dirs/test_output/RMIT_RMIT_test_test.ply --max-points 200000

    # Batch visualize all PLY files in a directory:
    python tools/visualize_ply.py work_dirs/test_output/ --batch

    # Color by instance_pred instead of semantic_pred:
    python tools/visualize_ply.py some.ply --color-by instance_pred

    # Color by ground truth:
    python tools/visualize_ply.py some.ply --color-by semantic_gt

Output: HTML file(s) next to the input PLY, openable in any browser.
No external dependencies required (uses plotly.js CDN).
"""
import argparse
import json
import os
import random
import sys

# Semantic class color map: ground=brown, wood=orange, leaf=green
SEMANTIC_COLORS = {
    0: (139, 119, 101),  # ground - brown
    1: (210, 150, 60),   # wood   - orange
    2: (34, 180, 34),    # leaf   - green
}
SEMANTIC_NAMES = {0: 'ground', 1: 'wood', 2: 'leaf'}

# Instance color palette (20 distinct colors, cycled for more instances)
INSTANCE_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (255, 187, 120), (152, 223, 138),
    (255, 152, 150), (197, 176, 213), (196, 156, 148), (247, 182, 210),
    (199, 199, 199), (219, 219, 141), (158, 218, 229), (57, 59, 121),
]


def parse_ply_ascii(filepath, max_points=100000):
    """Parse an ASCII PLY file. Returns (header_fields, rows) subsampled."""
    fields = []
    num_vertices = 0
    header_lines = 0

    with open(filepath, 'r') as f:
        for line in f:
            header_lines += 1
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                fields.append(parts[-1])  # field name
            elif line == 'end_header':
                break

        # Decide whether to subsample
        if num_vertices <= max_points:
            indices = None  # read all
        else:
            indices = set(sorted(random.sample(range(num_vertices), max_points)))

        rows = []
        for i, line in enumerate(f):
            if i >= num_vertices:
                break
            if indices is not None and i not in indices:
                continue
            vals = line.strip().split()
            rows.append(vals)

    return fields, rows, num_vertices


def build_html(fields, rows, total_points, color_by, title):
    """Build a standalone HTML string with an embedded plotly 3D scatter."""
    # Map field names to column indices
    col = {name: idx for idx, name in enumerate(fields)}

    xs, ys, zs = [], [], []
    colors = []
    hover_texts = []

    for row in rows:
        x = float(row[col['x']])
        y = float(row[col['y']])
        z = float(row[col['z']])
        xs.append(round(x, 3))
        ys.append(round(y, 3))
        zs.append(round(z, 3))

        sem_pred = int(row[col['semantic_pred']])
        inst_pred = int(row[col['instance_pred']])
        sem_gt = int(row[col['semantic_gt']])
        inst_gt = int(row[col['instance_gt']])

        # Color assignment
        if color_by == 'semantic_pred':
            c = SEMANTIC_COLORS.get(sem_pred, (128, 128, 128))
        elif color_by == 'semantic_gt':
            c = SEMANTIC_COLORS.get(sem_gt, (128, 128, 128))
        elif color_by == 'instance_pred':
            if inst_pred < 0:
                c = (128, 128, 128)
            else:
                c = INSTANCE_PALETTE[inst_pred % len(INSTANCE_PALETTE)]
        elif color_by == 'instance_gt':
            if inst_gt < 0:
                c = (128, 128, 128)
            else:
                c = INSTANCE_PALETTE[inst_gt % len(INSTANCE_PALETTE)]
        else:
            c = (128, 128, 128)

        colors.append(f'rgb({c[0]},{c[1]},{c[2]})')
        hover_texts.append(
            f'sem_pred={SEMANTIC_NAMES.get(sem_pred, sem_pred)}'
            f' inst_pred={inst_pred}'
            f'<br>sem_gt={SEMANTIC_NAMES.get(sem_gt, sem_gt)}'
            f' inst_gt={inst_gt}'
        )

    trace = {
        'type': 'scatter3d',
        'mode': 'markers',
        'x': xs,
        'y': ys,
        'z': zs,
        'text': hover_texts,
        'hoverinfo': 'text+x+y+z',
        'marker': {
            'size': 1.5,
            'color': colors,
            'opacity': 0.8,
        },
    }

    layout = {
        'title': f'{title}<br><sub>color={color_by} | '
                 f'{len(rows):,}/{total_points:,} points</sub>',
        'scene': {
            'xaxis': {'title': 'X'},
            'yaxis': {'title': 'Y'},
            'zaxis': {'title': 'Z'},
            'aspectmode': 'data',
        },
        'margin': {'l': 0, 'r': 0, 'b': 0, 't': 60},
    }

    # Legend entries for semantic coloring
    legend_traces = []
    if color_by.startswith('semantic'):
        for cls_id, cls_name in SEMANTIC_NAMES.items():
            c = SEMANTIC_COLORS[cls_id]
            legend_traces.append({
                'type': 'scatter3d',
                'mode': 'markers',
                'x': [None], 'y': [None], 'z': [None],
                'name': cls_name,
                'marker': {'size': 8, 'color': f'rgb({c[0]},{c[1]},{c[2]})'},
                'showlegend': True,
            })

    data_json = json.dumps([trace] + legend_traces)
    layout_json = json.dumps(layout)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #plot {{ width: 100vw; height: 100vh; }}
    </style>
</head>
<body>
    <div id="plot"></div>
    <script>
        var data = {data_json};
        var layout = {layout_json};
        Plotly.newPlot('plot', data, layout, {{responsive: true}});
    </script>
</body>
</html>"""
    return html


def process_file(ply_path, output_dir, max_points, color_by):
    """Process a single PLY file and write HTML."""
    basename = os.path.splitext(os.path.basename(ply_path))[0]
    title = basename
    print(f'Processing: {basename} ...', end=' ', flush=True)

    fields, rows, total = parse_ply_ascii(ply_path, max_points)
    print(f'{len(rows):,}/{total:,} points loaded.', end=' ', flush=True)

    html = build_html(fields, rows, total, color_by, title)

    out_path = os.path.join(output_dir, f'{basename}_{color_by}.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'-> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ForestFormer3D PLY predictions as HTML')
    parser.add_argument('input', help='PLY file or directory of PLY files')
    parser.add_argument('--batch', action='store_true',
                        help='Process all .ply files in the input directory')
    parser.add_argument('--max-points', type=int, default=100000,
                        help='Max points to display (default: 100000)')
    parser.add_argument('--color-by', default='semantic_pred',
                        choices=['semantic_pred', 'semantic_gt',
                                 'instance_pred', 'instance_gt'],
                        help='Field to color by (default: semantic_pred)')
    parser.add_argument('--output-dir',
                        help='Output directory (default: same as input)')
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        ply_dir = args.input
        ply_files = sorted([
            os.path.join(ply_dir, f) for f in os.listdir(ply_dir)
            if f.endswith('.ply')
        ])
        if not ply_files:
            print(f'No .ply files found in {ply_dir}')
            sys.exit(1)
        print(f'Found {len(ply_files)} PLY files.')
    else:
        ply_files = [args.input]

    output_dir = args.output_dir or (
        args.input if os.path.isdir(args.input)
        else os.path.dirname(args.input) or '.'
    )
    os.makedirs(output_dir, exist_ok=True)

    for ply_path in ply_files:
        process_file(ply_path, output_dir, args.max_points, args.color_by)

    print('Done!')


if __name__ == '__main__':
    main()
