import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Directory setup
models_dir = 'models/trained_model/'

# Load models
generator = load_model(os.path.join(models_dir, 'generator.h5'))

# Parameters
latent_dim = 100

# Generate and visualize images
def generate_images(generator, num_images):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_images = generator.predict(noise)
    return gen_images

num_images = 10
gen_images = generate_images(generator, num_images)

plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    plt.imshow((gen_images[i] * 127.5 + 127.5).astype(np.uint8))
    plt.axis('off')
plt.show()