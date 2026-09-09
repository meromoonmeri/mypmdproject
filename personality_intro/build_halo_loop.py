"""
DREAM_HALO_RAINBOW_LOOP — halo arc-en-ciel animé, bouclage exact.

Couches indépendantes, chacune exportée séparément (PNG RGBA sur fond noir
transparent) pour être recomposée dans le moteur :

  ring_01  anneaux arc-en-ciel, rotation lente horaire      (+1 tour/boucle)
  ring_02  anneaux arc-en-ciel, rotation inverse            (-2 tours/boucle)
  rays     rayons fins, rotation lente                      (+1 tour/boucle)
  core     coeur sombre + halo blanc central, pulsation
  composite  les couches déjà fusionnées (aperçu / usage direct)

L'ARC-EN-CIEL EST ANIMÉ : la teinte défile le long du rayon. Comme le
défilement est un nombre ENTIER de cycles par boucle, la dernière frame se
raccorde exactement à la première.

Règle de bouclage : tout terme temporel est soit un entier de tours, soit
un cos/sin de 2*pi*k*phase avec k entier.
"""
from __future__ import annotations
import json, os
import numpy as np
from PIL import Image

TAU = 2.0 * np.pi
ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
FRAMES = 36
FPS = 12
HUE_CYCLES = 1     # cycles de defilement de teinte par boucle (ENTIER)


def polar(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx) / (min(h, w) / 2.0)
    return r.astype(np.float32), np.arctan2(dy, dx).astype(np.float32)


def polar_strip(path, n_theta=1024, n_rad=512):
    """Echantillonne une plaque carree sur des cercles complets.
    La bande obtenue est cycliquement continue en theta : la rotation ne
    peut donc jamais creer de couture."""
    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = im.shape[:2]
    th = np.linspace(0, TAU, n_theta, endpoint=False, dtype=np.float32)
    rr = np.linspace(0, 1, n_rad, dtype=np.float32)
    R, T = np.meshgrid(rr, th, indexing="ij")
    y = np.clip((h / 2 - 1) * (1 + R * np.sin(T)), 0, h - 1).astype(np.int32)
    x = np.clip((w / 2 - 1) * (1 + R * np.cos(T)), 0, w - 1).astype(np.int32)
    return im[y, x]


def sample(strip, r, th, spin_turns, phase):
    n_rad, n_theta = strip.shape[:2]
    ti = ((th / TAU + spin_turns * phase) * n_theta)
    ti = np.mod(ti.astype(np.int64), n_theta).astype(np.int32)   # entier PUIS modulo
    ri = np.clip(r * (n_rad - 1), 0, n_rad - 1).astype(np.int32)
    return strip[ri, ti]


def hue_shift(rgb, amount):
    """Rotation de teinte vectorisee (matrice YIQ). `amount` en tours."""
    a = amount * TAU
    c, s = np.cos(a), np.sin(a)
    m = np.array([
        [0.299 + 0.701 * c + 0.168 * s, 0.587 - 0.587 * c + 0.330 * s, 0.114 - 0.114 * c - 0.497 * s],
        [0.299 - 0.299 * c - 0.328 * s, 0.587 + 0.413 * c + 0.035 * s, 0.114 - 0.114 * c + 0.292 * s],
        [0.299 - 0.300 * c + 1.250 * s, 0.587 - 0.588 * c - 1.050 * s, 0.114 + 0.886 * c - 0.203 * s],
    ], np.float32)
    return np.clip(rgb @ m.T, 0, 1)


def to_rgba(rgb):
    """Fond noir -> alpha. Le halo doit se superposer aux etoiles."""
    a = np.clip(rgb.max(2) * 1.35, 0, 1)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def main():
    rings = polar_strip(os.path.join(ROOT, "dream_halo/plate_rainbow_rings.png"))
    rays = polar_strip(os.path.join(ROOT, "dream_halo/plate_rays.png"))
    r, th = polar(H, W)
    # rayon normalise sur la diagonale : le halo deborde de l'ecran (PHASE 04)
    rd = (r / 1.35).astype(np.float32)

    layers = {
        "ring_01": dict(strip=rings, spin=+1, scale=1.00, gain=1.00),
        "ring_02": dict(strip=rings, spin=-2, scale=0.62, gain=0.85),
        "rays":    dict(strip=rays,  spin=+1, scale=1.00, gain=0.70),
    }
    dirs = {k: os.path.join(ROOT, "frames", f"dream_halo_{k}") for k in layers}
    dirs["core"] = os.path.join(ROOT, "frames", "dream_halo_core")
    dirs["composite"] = os.path.join(ROOT, "frames", "dream_halo_composite")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # coeur sombre central, comme sur la reference
    core_dark = np.clip((rd / 0.20) ** 1.6, 0, 1)[..., None]

    for i in range(FRAMES):
        ph = i / FRAMES
        comp = np.zeros((H, W, 3), np.float32)

        for name, L in layers.items():
            rgb = sample(L["strip"], np.clip(rd / L["scale"], 0, 1), th, L["spin"], ph)
            if name.startswith("ring"):
                # ARC-EN-CIEL ANIME : la teinte defile le long du rayon.
                # entier de cycles => raccord exact de la derniere frame.
                rgb = hue_shift(rgb, HUE_CYCLES * ph)
            rgb = rgb * L["gain"] * core_dark
            to_rgba(rgb).save(os.path.join(dirs[name], f"frame_{i:03d}.png"), optimize=True)
            comp = 1.0 - (1.0 - comp) * (1.0 - rgb)      # screen : lumiere additive

        # coeur : halo blanc central pulsant (1 cycle entier par boucle)
        pulse = 0.5 + 0.5 * np.cos(TAU * ph)
        glow = np.exp(-(rd / (0.17 + 0.03 * pulse)) ** 2)[..., None]
        core = glow * np.array([1.0, 0.99, 0.96], np.float32) * (0.55 + 0.45 * pulse)
        to_rgba(core).save(os.path.join(dirs["core"], f"frame_{i:03d}.png"), optimize=True)
        comp = 1.0 - (1.0 - comp) * (1.0 - core)

        Image.fromarray(np.clip(comp * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(dirs["composite"], f"frame_{i:03d}.png"), optimize=True)

    meta = {
        "nom": "DREAM_HALO_RAINBOW_LOOP",
        "type": "boucle cyclique exacte (pas de ping-pong necessaire)",
        "frames": FRAMES, "fps": FPS, "duree_s": round(FRAMES / FPS, 3),
        "frame_depart": 0, "frame_fin": FRAMES - 1,
        "point_de_boucle": f"frame_{FRAMES-1:03d} -> frame_000 (raccord exact, phase 1.0 == phase 0.0)",
        "arc_en_ciel_anime": {
            "methode": "rotation de teinte YIQ le long du rayon",
            "cycles_par_boucle": HUE_CYCLES,
            "raccord": "entier de cycles => derniere frame identique a la premiere",
        },
        "couches": [
            {"nom": "ring_01", "couche": "ANNEAU 01", "rotation": "+1 tour/boucle (horaire, lente)",
             "scale": 1.00, "opacite": 1.00, "parallaxe": 0.30, "effet": "arc-en-ciel defilant"},
            {"nom": "ring_02", "couche": "ANNEAU 02", "rotation": "-2 tours/boucle (inverse)",
             "scale": 0.62, "opacite": 0.85, "parallaxe": 0.50, "effet": "arc-en-ciel defilant"},
            {"nom": "rays", "couche": "ANNEAU 03", "rotation": "+1 tour/boucle",
             "scale": 1.00, "opacite": 0.70, "parallaxe": 0.40, "effet": "rayons fins"},
            {"nom": "core", "couche": "ANNEAU 06 - halo blanc central", "rotation": 0,
             "scale": 1.00, "opacite": "0.55 -> 1.00 pulsation", "parallaxe": 0.60,
             "effet": "pulsation lumineuse, 1 cycle par boucle"},
        ],
        "position": "centre, deborde volontairement de l'ecran",
        "transition": "declenchee par le dialogue uniquement, jamais par timer",
        "resolution": f"{W}x{H}",
    }
    json.dump(meta, open(os.path.join(ROOT, "frames", "dream_halo_loop.json"), "w"),
              indent=2, ensure_ascii=False)
    print("[OK]", FRAMES, "frames x", len(dirs), "couches")


if __name__ == "__main__":
    main()
