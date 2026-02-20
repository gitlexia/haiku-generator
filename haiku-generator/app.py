import os
from flask import Flask, request, jsonify, send_from_directory

# IMPORTANT: your generator file must be named haiku_generator.py
# and must contain a function: generate_best_haiku(prompt: str) -> str
from haiku_generator import generate_best_haiku

app = Flask(__name__)

# Serve the front-end page
@app.get("/")
def home():
    return send_from_directory(".", "index.html")

# Optional: serve other static files if you add them later (css, js, etc.)
@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    haiku = generate_best_haiku(prompt)
    return jsonify({"haiku": haiku})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)