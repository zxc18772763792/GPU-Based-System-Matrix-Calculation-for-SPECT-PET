#!/usr/bin/env python3
"""
Serve the SPECT/PET element viewer and a data directory (parameter files,
system matrices, Monte Carlo outputs) to a browser.

Only the Python standard library is used. System matrix files are never loaded
into memory: the viewer reads rows, columns and chunks through HTTP Range
requests and the /api/strided endpoint.

    python3 serve.py --data /path/to/run_dir            # then open http://127.0.0.1:8765
    python3 serve.py --data . --port 9000 --open

On a remote GPU server keep the default 127.0.0.1 binding and use SSH port
forwarding:   ssh -L 8765:127.0.0.1:8765 user@gpu-server
"""
import argparse
import http.server
import json
import mimetypes
import os
import re
import socketserver
import sys
import urllib.parse
import webbrowser

UI_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_EXT = (".dat", ".sysmat", ".csv", ".json", ".bin", ".raw", ".img", ".txt")
MAX_STRIDED = 1 << 22  # values per /api/strided request


LOCAL_NAMES = {"127.0.0.1", "localhost", "::1"}


class Handler(http.server.SimpleHTTPRequestHandler):
    data_dir = "."
    allowed_hosts = LOCAL_NAMES   # Host header names accepted (None: any); blocks DNS rebinding of a local server

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=UI_DIR, **kw)

    def log_message(self, fmt, *args):  # quieter log
        if os.environ.get("SV_VERBOSE"):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ------------------------------------------------------------ helpers
    def _host_ok(self):
        if self.allowed_hosts is None:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):                      # [::1]:port
            host = host[1:host.find("]")] if "]" in host else host
        else:
            host = host.rsplit(":", 1)[0]
        return host in self.allowed_hosts

    def _data_path(self, rel):
        """rel: path relative to the data directory, already URL-decoded"""
        root = os.path.realpath(self.data_dir)
        p = os.path.realpath(os.path.join(root, rel))
        if os.path.commonpath([root, p]) != root or not os.path.isfile(p):
            return None
        return p

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    # ------------------------------------------------------------ routes
    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/files":
            return self._files()
        if url.path == "/api/strided":
            return self._strided(urllib.parse.parse_qs(url.query))
        if url.path.startswith("/data/"):
            return self._data(urllib.parse.unquote(url.path[len("/data/"):]), head=False)
        return super().do_GET()

    def do_HEAD(self):
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urllib.parse.urlparse(self.path)
        if url.path.startswith("/data/"):
            return self._data(urllib.parse.unquote(url.path[len("/data/"):]), head=True)
        return super().do_HEAD()

    def _files(self):
        root = os.path.realpath(self.data_dir)
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            depth = os.path.relpath(dirpath, root).count(os.sep)
            dirnames[:] = [d for d in sorted(dirnames) if not d.startswith(".")] if depth < 2 else []
            for fn in sorted(filenames):
                if fn.lower().endswith(DATA_EXT):
                    p = os.path.join(dirpath, fn)
                    real = os.path.realpath(p)
                    if os.path.commonpath([root, real]) != root:   # symlink leaving the data directory
                        continue
                    try:
                        size = os.path.getsize(real)
                    except OSError:                                # dangling link, file removed meanwhile
                        continue
                    out.append({"path": os.path.relpath(p, root).replace(os.sep, "/"), "size": size})
        self._json({"root": root, "files": out})

    def _data(self, rel, head):
        p = self._data_path(rel)
        if not p:
            return self._error(404, "not found")
        size = os.path.getsize(p)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        # single byte range; anything else (other units, multiple ranges, malformed) is ignored -> 200 (RFC 9110)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range: the last N bytes
                start = max(0, size - int(m.group(2)))
            if end >= start:                              # "bytes=10-5" is invalid and ignored
                if start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                partial = True
            else:
                start, end = 0, size - 1
        length = end - start + 1
        self.send_response(206 if partial else 200)
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _strided(self, q):
        """count float32 values at byte offset + i*stride (e.g. one voxel column of a sysmat)."""
        try:
            p = self._data_path(q["file"][0])
            offset, stride, count = int(q["offset"][0]), int(q["stride"][0]), int(q["count"][0])
        except (KeyError, ValueError):
            return self._error(400, "need file, offset, stride, count")
        if not p:
            return self._error(404, "not found")
        if count <= 0 or count > MAX_STRIDED or stride < 4 or offset < 0:
            return self._error(400, "bad request")
        size = os.path.getsize(p)
        if offset + (count - 1) * stride + 4 > size:
            return self._error(416, "out of range")
        out = bytearray(count * 4)
        with open(p, "rb") as f:
            for i in range(count):
                f.seek(offset + i * stride)
                out[i * 4:i * 4 + 4] = f.read(4)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(bytes(out))


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".", help="directory with Params_*.dat / *.sysmat / MC outputs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open a browser tab")
    args = ap.parse_args()
    if not os.path.isdir(args.data):
        sys.exit(f"not a directory: {args.data}")
    Handler.data_dir = os.path.abspath(args.data)
    # requests must name this server (blocks DNS rebinding); on a non-local interface any name is accepted
    Handler.allowed_hosts = LOCAL_NAMES if args.host in LOCAL_NAMES else None
    srv = Server((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"Serving viewer at {url}  (data: {Handler.data_dir})  - Ctrl+C to stop")
    if args.host not in ("127.0.0.1", "localhost"):
        print("WARNING: listening on a non-local interface; anyone who can reach this port can read the data directory")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
