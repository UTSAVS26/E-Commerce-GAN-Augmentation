from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
models_dir = 'models/trained_model/'
generator = load_model(f'{models_dir}/generator.h5')

# Parameters
latent_dim = 100

@app.route('/generate', methods=['POST'])
def generate():
    num_images = request.json.get('num_images', 1)
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = (gen_images * 127.5 + 127.5).astype(np.uint8)
    return jsonify(gen_images.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)