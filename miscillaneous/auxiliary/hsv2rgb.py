import numpy as np

def color_fun(H):
    result = 0
    if 0 <= H < 60:
        result = H / 60
    elif 60 <= H < 180:
        result = 1
    elif 240 <= H <= 360:
        result = 0
    elif 180 <= H < 240:
        result = 4 - H / 60
    return result

def hsv2rgb(n, in_v, in_h):
    nom = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) + np.finfo(float).eps
    F = np.arctan2(n[1] / nom, n[0] / nom)
    H = 360 * in_h + (1 - 2 * in_h) * (F if F >= 0 else 2 * np.pi + F) * 180 / np.pi
    H = H % 360

    m1 = 1 - abs(n[2]) / nom if (1 - 2 * in_v) * n[2] / nom < 0 else 1
    m2 = 0 if (1 - 2 * in_v) * n[2] / nom < 0 else abs(n[2]) / nom

    max_v = 0.5 + nom * (m1 - 0.5)
    min_v = 0.5 - nom * (0.5 - m2)
    dV = max_v - min_v

    rgb = list(n)
    rgb[0] = np.round(color_fun((H + 120) % 360) * dV + min_v, decimals=10)
    rgb[1] = np.round(color_fun(H % 360) * dV + min_v, decimals=10)
    rgb[2] = np.round(color_fun((H - 120) % 360) * dV + min_v, decimals=10)

    return rgb