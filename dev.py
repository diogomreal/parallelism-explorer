#!/usr/bin/env python3
"""Dev server for site/ that disables browser caching, so edits show up on a normal reload.
Usage: python3 dev.py [port]   (default 8000)"""
import functools, http.server, os, sys

class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site')
print(f'Serving {root} at http://localhost:{port}')
http.server.ThreadingHTTPServer(('', port), functools.partial(NoCache, directory=root)).serve_forever()
