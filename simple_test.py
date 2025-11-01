#!/usr/bin/env python3
"""
Simple test app to verify basic Flask functionality
"""

from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/')
def root():
    return jsonify({
        "status": "ok",
        "service": "Reconciliation API Test",
        "port": os.environ.get('PORT', 'not_set')
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "test": True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"Starting simple test app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)