
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def cwd() -> Path:
    return Path(__file__).resolve().parent.parent


def save_img(img: np.ndarray, path: Path) -> None:
    plt.imsave(path, img)