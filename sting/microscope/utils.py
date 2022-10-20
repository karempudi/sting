import numpy as np

def fetch_mm_image():
    # fetches the current micromanger image
    return np.random.randint(low = 0, high = 2, size=(10, 10))