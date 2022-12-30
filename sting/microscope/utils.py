import numpy as np
from pycromanager import Core
import sys

def fetch_mm_image():
    # fetches the current micromanger image
    #return np.random.randint(low = 0, high = 2, size=(10, 10))
    try:
        core = Core()
        core.snap_image()
        tagged_image = core.get_tagged_image()
        pixels = np.reshape(tagged_image.pix,
                newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        return pixels
    except Exception as e:
        sys.stdout.write(f"Error {e} while snapping image on the microscope ... \n")
        sys.stdout.flush()
        return np.random.randint(low = 0, high = 2, size=(10, 10))
    