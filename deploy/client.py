#!/usr/bin/env python3
"""ForestFormer3D API Client.

Simple client for the deployed ForestFormer3D API.

Usage:
    python deploy/client.py upload forest_scan.las
    python deploy/client.py upload forest_scan.las --server http://gpu-server:8000
    python deploy/client.py status
    python deploy/client.py download <task_id> -o result.ply
"""
import argparse
import json
import os
import sys

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


def upload(args):
    """Upload a point cloud file for segmentation."""
    url = f"{args.server}/predict"
    params = {}
    if args.subsample:
        params['subsample'] = args.subsample
    if args.max_points:
        params['max_points'] = args.max_points

    filename = os.path.basename(args.file)
    print(f"Uploading {filename} to {args.server} ...")
    with open(args.file, 'rb') as f:
        resp = requests.post(url, files={'file': (filename, f)}, params=params,
                             timeout=3600)

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    print(f"\nTask: {data['task_id']}")
    print(f"Status: {data['status']}")
    if 'stats' in data:
        stats = data['stats']
        print(f"\nResults:")
        print(f"  Points processed: {stats.get('n_processed', '?'):,}")
        print(f"  Trees detected:   {stats.get('n_trees', '?')}")
        print(f"  Inference time:   {stats.get('inference_time_s', '?')}s")
        if 'semantic' in stats:
            s = stats['semantic']
            print(f"  Semantic: ground={s['ground']:,} wood={s['wood']:,} leaf={s['leaf']:,}")
    if 'download_url' in data:
        print(f"\nDownload: {args.server}{data['download_url']}")

        # Auto-download if -o specified
        if args.output:
            print(f"Downloading to {args.output} ...")
            r = requests.get(f"{args.server}{data['download_url']}", timeout=300)
            with open(args.output, 'wb') as f:
                f.write(r.content)
            print(f"Saved: {args.output} ({len(r.content)/1e6:.1f} MB)")


def status(args):
    """Check server status."""
    resp = requests.get(f"{args.server}/health", timeout=10)
    print(json.dumps(resp.json(), indent=2))


def tasks(args):
    """List all tasks."""
    resp = requests.get(f"{args.server}/tasks", timeout=10)
    data = resp.json()
    if not data:
        print("No tasks.")
        return
    for tid, info in data.items():
        print(f"  {tid}: {info.get('status', '?')} - {info.get('filename', '?')}")


def download(args):
    """Download result PLY."""
    url = f"{args.server}/result/{args.task_id}/ply"
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=300)
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return
    output = args.output or f"{args.task_id}_result.ply"
    with open(output, 'wb') as f:
        f.write(resp.content)
    print(f"Saved: {output} ({len(resp.content)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='ForestFormer3D API Client')
    parser.add_argument('--server', default='http://localhost:8000',
                        help='Server URL (default: http://localhost:8000)')
    sub = parser.add_subparsers(dest='cmd')

    p_upload = sub.add_parser('upload', help='Upload file for segmentation')
    p_upload.add_argument('file', help='LAS/LAZ/PLY file')
    p_upload.add_argument('-o', '--output', help='Auto-download result to this path')
    p_upload.add_argument('--subsample', type=float, default=0.05)
    p_upload.add_argument('--max-points', type=int, default=500000)

    p_status = sub.add_parser('status', help='Server health check')
    p_tasks = sub.add_parser('tasks', help='List tasks')

    p_dl = sub.add_parser('download', help='Download result')
    p_dl.add_argument('task_id')
    p_dl.add_argument('-o', '--output')

    args = parser.parse_args()
    if args.cmd == 'upload':
        upload(args)
    elif args.cmd == 'status':
        status(args)
    elif args.cmd == 'tasks':
        tasks(args)
    elif args.cmd == 'download':
        download(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
