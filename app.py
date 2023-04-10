import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, request, jsonify, render_template
import atexit
import zmq
import socket

# Step 1: Set up the environment
#pip install transformers torch flask

# Step 2: Load the GPT-Neo model
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Workaround for CUDA out of memory error
if device == 'cuda':
    model.half()
    for layer in model.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.float()
    torch.cuda.empty_cache()

model.to(device)

# This function finds an available port and returns it.

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

open_port = get_open_port()

# Step 3: Define the Flask app
app = Flask(__name__)

# Define ZeroMQ socket to receive stop signal
stop_signal = zmq.Context().socket(zmq.PULL)
stop_signal.bind(f"tcp://127.0.0.1:{open_port}")

# Define function to gracefully shutdown the Flask app
def shutdown_server():
    print('Stopping the Flask app...')
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug server')
    func()

# Call the shutdown function on ZeroMQ stop signal
def stop_server():
    stop_signal.recv()
    shutdown_server()

# Register the shutdown function with Flask app on exit
atexit.register(shutdown_server)

# Define the root route
@app.route('/')
def index():
    return render_template('pages/index.html')

@app.route('/qa', methods=['POST'])
def qa():
   # query = request.json['query']
   # answer = get_answer(query)
   # Get the user's question from the form
    query = request.form['question']
    
    # Get the answer using the get_answer function
    answer = get_answer(query)
    # Return the answer as a JSON response
    return jsonify({'answer': answer})

# Step 4: Define the answer retrieval function
def get_answer(query):
    with torch.cuda.amp.autocast():
        inputs = tokenizer.encode("Q: " + query + "\nA:", return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=1024, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("A:")[1].strip()

    # Fix for CUDA out of memory error
    if device == 'cuda':
        torch.cuda.empty_cache()

    return answer

# Start a background thread to listen for the stop signal
import threading
stop_thread = threading.Thread(target=stop_server)
stop_thread.start()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

