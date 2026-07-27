from pathlib import Path
from PIL import Image
import numpy as np

def read_image(path: Path | str) -> np.ndarray:
    img = Image.open(path)
    a = np.asarray(img)
    print(a.shape)
    return a
