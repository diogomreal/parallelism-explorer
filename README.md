# Parallelism Explorer

Web tool for exploring how TP / PP / DP / EP / CP choices change LLM inference throughput,
latency and cost on an NVL72 rack.

## Run locally

    python3 dev.py                          # then open http://localhost:8000 (sends no-store, so edits show on reload)
    python3 -m http.server -d site 8000     # also works, but browsers may cache the JS modules: hard-reload (Ctrl+Shift+R)
