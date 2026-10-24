#!/usr/bin/env python3
"""
generate_assets.py  --  builds every visual asset for the shalabh.exe README.

Outputs
  portrait.txt           ASCII-art portrait (plain text, copy-paste friendly)
  assets/portrait.png    same portrait rendered as neon-green glyphs on black
  assets/matrix.gif      looping Matrix-rain banner with a glitching title
  assets/intro.svg       typing-animation terminal intro (SMIL, GitHub-safe)
  assets/footer.svg      glitching "[ CONNECTION TERMINATED ]" footer

Usage
  pip install -r requirements.txt
  python generate_assets.py

Only Pillow + NumPy are needed. Fonts are searched on the system; if none of
the preferred ones exist the script falls back to Pillow's built-in font.
"""

import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SOURCE = ROOT / "source_portrait.png"   # <- replace with another photo if you like
NEON = (0, 255, 102)                    # #00FF66
BLACK = (0, 0, 0)
GRAY = (139, 148, 158)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def find_font(candidates, size, index=0):
    """Return the first font in `candidates` that exists, else Pillow default."""
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default()

MONO = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "C:/Windows/Fonts/consolab.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]
MONO_REG = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]
KANA = [  # any font with half-width katakana; Matrix rain falls back to hex if missing
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

# --------------------------------------------------------------------------
# 1) ASCII portrait
# --------------------------------------------------------------------------
RAMP = " .'`,:;-~=+*xoa#%&8@"   # light -> dense (20 levels for finer detail)
COLS = 104                   # characters per row
CROP = (0.27, 0.00, 0.73, 0.60)  # (left, top, right, bottom) as fractions of the image

def build_portrait():
    img = Image.open(SOURCE).convert("RGB")
    w, h = img.size
    # crop to head + collar (fractions of the image)
    box = (int(w * CROP[0]), int(h * CROP[1]), int(w * CROP[2]), int(h * CROP[3]))
    img = img.crop(box)
    cw, ch = img.size

    # background mask: the backdrop is a flat purple, so colour distance works
    bg = np.array(img.getpixel((4, 4)), dtype=float)
    arr = np.asarray(img, dtype=float)
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = Image.fromarray(((dist > 38) * 255).astype("uint8")).filter(ImageFilter.MedianFilter(7))

    # luminance, boosted local contrast + a touch of sharpening for facial detail
    gray = ImageOps.autocontrast(img.convert("L"), cutoff=0.5)
    lum = Image.blend(gray, ImageOps.equalize(gray), 0.35)      # lift local detail
    lum = lum.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))

    rows = int(COLS * (ch / cw) * 0.5)          # glyphs are ~2x taller than wide
    lum_s = np.asarray(lum.resize((COLS, rows), Image.LANCZOS), dtype=float) / 255
    mask_s = np.asarray(mask.resize((COLS, rows), Image.BILINEAR), dtype=float) / 255

    # gamma < 1 opens up the shadows so hair strands show texture instead of a solid blob
    dark = 1.0 - lum_s ** 0.72                  # hair/glasses/beard = dense glyphs
    dark = np.clip((dark - 0.10) / 0.80, 0, 1)

    # edge pass: draws glasses frames, nose, lips and jawline as crisp strokes
    # (the source is pixel art, so blur first or every pixel block becomes an "edge")
    soft = lum.filter(ImageFilter.GaussianBlur(cw / COLS * 1.6))
    big = np.asarray(soft.resize((COLS * 2, rows * 2), Image.LANCZOS), dtype=float) / 255
    gy, gx = np.gradient(big)
    edge = np.hypot(gx, gy)
    edge = np.asarray(Image.fromarray((np.clip((edge - 0.035) * 9, 0, 1) * 255).astype("uint8"))
                      .resize((COLS, rows), Image.BOX), dtype=float) / 255
    dark = np.clip(np.maximum(dark, edge * 0.85), 0, 1)

    lines, density = [], []
    for y in range(rows):
        row, drow = "", []
        for x in range(COLS):
            if mask_s[y, x] < 0.5:
                row += " "; drow.append(0.0)
            else:
                d = dark[y, x]
                row += RAMP[min(int(d * len(RAMP)), len(RAMP) - 1)]
                drow.append(d)
        lines.append(row.rstrip()); density.append(drow)
    return lines, density

def save_portrait(lines, density):
    (ROOT / "portrait.txt").write_text("\n".join(lines).strip("\n") + "\n", encoding="utf-8")

    scale = 11                                    # font px; cell = 0.6*scale x 1.2*scale
    font = find_font(MONO_REG, scale * 2)
    cw, chh = int(scale * 1.2), int(scale * 2.4)  # supersampled 2x for crispness
    W, H = cw * COLS + 40, chh * len(lines) + 40
    im = Image.new("RGB", (W, H), BLACK)
    dr = ImageDraw.Draw(im)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == " ":
                continue
            d = density[y][x]
            # dim green for light strokes -> full neon for dense ones
            c = tuple(int(a + (b - a) * min(1, d * 1.15)) for a, b in zip((0, 90, 40), NEON))
            dr.text((20 + x * cw, 20 + y * chh), ch, font=font, fill=c)
    # faint scanlines for the CRT mood
    sl = ImageDraw.Draw(im, "RGBA")
    for y in range(0, H, 4):
        sl.line([(0, y), (W, y)], fill=(0, 0, 0, 70))
    # palette-reduce: the image is only greens on black, so 48 colours is plenty and ~3x smaller
    im.quantize(colors=48, dither=Image.Dither.NONE).save(ASSETS / "portrait.png", optimize=True)

# --------------------------------------------------------------------------
# 2) Matrix rain GIF (seamless loop)
# --------------------------------------------------------------------------
GW, GH = 900, 220
CELL_W, CELL_H = 14, 18
FRAMES = 48           # loop length
CYCLE = 24            # rows per rain cycle; speeds below divide FRAMES*speed evenly
TRAIL = 9
FPS_MS = 70

def glyph_pool():
    kana = [chr(c) for c in range(0xFF66, 0xFF9E)]
    other = list("0123456789ABCDEF:=*+<>|")
    return kana + other

def draw_title(size_main=76):
    """Returns (backing, lettering, (x, y, width)); lettering is what gets glitched."""
    back = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))
    layer = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = find_font(MONO, size_main)
    f2 = find_font(MONO_REG, 17)
    text = "SHALABH.EXE"
    tw = d.textlength(text, font=f)
    x, y = (GW - tw) / 2, GH / 2 - 62
    ImageDraw.Draw(back).rectangle([x - 22, y - 8, x + tw + 22, y + size_main + 62], fill=(0, 0, 0, 205))
    d.text((x, y), text, font=f, fill=NEON + (255,))
    sub = "// B.Tech CSE  ·  Raipur, IN  ·  builder"
    sw = d.textlength(sub, font=f2)
    d.text(((GW - sw) / 2, y + size_main + 14), sub, font=f2, fill=GRAY + (255,))
    return back, layer, (x, y, tw)

def tinted(layer, color):
    r, g, b, a = layer.split()
    solid = Image.new("RGBA", layer.size, color + (0,))
    solid.putalpha(a.point(lambda v: int(v * 0.85)))
    return solid

def make_matrix_gif():
    rnd = random.Random(1337)
    pool = glyph_pool()
    cols, rows = GW // CELL_W, GH // CELL_H + 1
    kana_font = find_font(KANA, 15)
    # per-column stream settings
    streams = [dict(off=rnd.randrange(CYCLE), speed=rnd.choice([0.5, 1.0, 1.0, 0.5, 2.0 / 1.0 * 0.5]))
               for _ in range(cols)]
    # glyph table: [col][row][flicker-step]
    table = [[[rnd.choice(pool) for _ in range(FRAMES // 6)] for _ in range(rows)] for _ in range(cols)]

    back, title, (tx, ty, tw) = draw_title()
    glitch_frames = {9, 10, 11, 29, 30, 40}
    frames = []
    for t in range(FRAMES):
        im = Image.new("RGB", (GW, GH), BLACK)
        dr = ImageDraw.Draw(im)
        for c in range(cols):
            s = streams[c]
            head = (s["off"] + s["speed"] * t) % CYCLE
            for r in range(rows):
                dist = (head - r) % CYCLE
                if dist >= TRAIL:
                    continue
                g = table[c][r][(t // 6) % (FRAMES // 6)]
                if dist < 1:   fill = (200, 255, 220)             # bright head
                else:          k = 1 - dist / TRAIL; fill = (0, int(40 + 215 * k * k), int(15 + 70 * k * k))
                dr.text((c * CELL_W, r * CELL_H), g, font=kana_font, fill=fill)

        im.paste(back, (0, 0), back)
        if t in glitch_frames:   # RGB-split + sliced title
            jitter = rnd.choice([-1, 1]) * rnd.randint(4, 7)
            im.paste(tinted(title, (255, 40, 80)), (-jitter, 0), tinted(title, (255, 40, 80)))
            im.paste(tinted(title, (0, 220, 255)), (jitter, 0), tinted(title, (0, 220, 255)))
            sliced = title.copy()
            for _ in range(3):
                y0 = int(ty + rnd.randint(0, 70)); hh = rnd.randint(6, 16)
                band = title.crop((0, y0, GW, y0 + hh))
                sliced.paste(Image.new("RGBA", (GW, hh), (0, 0, 0, 0)), (0, y0))
                sliced.paste(band, (rnd.randint(-26, 26), y0))
            im.paste(sliced, (0, 0), sliced)
        else:
            im.paste(title, (0, 0), title)

        # scanlines + vignette-ish edge fade
        sl = ImageDraw.Draw(im, "RGBA")
        for y in range(0, GH, 3):
            sl.line([(0, y), (GW, y)], fill=(0, 0, 0, 55))
        frames.append(im)

    # one shared palette keeps the GIF small and flicker-free
    pal_src = Image.new("RGB", (GW, GH * 3))
    for i, k in enumerate((0, 10, 30)):
        pal_src.paste(frames[k], (0, GH * i))
    pal = pal_src.quantize(colors=36, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(ASSETS / "matrix.gif", save_all=True, append_images=q[1:],
              duration=FPS_MS, loop=0, optimize=True, disposal=1)

# --------------------------------------------------------------------------
# 3) Typing-intro SVG (pure SMIL, no JS, GitHub-safe)
# --------------------------------------------------------------------------
FONT_STACK = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono','DejaVu Sans Mono',monospace"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def make_intro_svg():
    # (segments of (text, colour)) per line
    G, W, D, A = "#00FF66", "#e6edf3", "#8b949e", "#ffffff"
    lines = [
        [("> ", G), ("shalabh.exe --wake", W)],
        [("[ OK ] ", G), ("human: raipur, IN", D)],
        [("[ OK ] ", G), ("B.Tech CSE // builder", D)],
        [("[WARN] ", A), ("sleep.exe not found", D)],
        [("> ", G), ("hello, world.", W)],
    ]
    cw, fs, lh = 9.0, 15, 27           # char width, font size, line height
    pad_x, pad_y = 18, 34
    widest = max(sum(len(t) for t, _ in segs) for segs in lines)
    width = int(pad_x * 2 + cw * max(widest, 26))
    height = pad_y + lh * len(lines) + 16
    loop = 15.0                        # seconds for the full cycle
    t0, per_line, gap = 0.6, 1.5, 0.45 # start, typing time, pause

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="Terminal intro: shalabh.exe wake sequence">',
           f'<rect width="100%" height="100%" rx="10" fill="#000" stroke="#00FF66" stroke-opacity=".55"/>',
           '<circle cx="18" cy="16" r="4" fill="#00FF66" fill-opacity=".9"/>'
           '<circle cx="32" cy="16" r="4" fill="#00FF66" fill-opacity=".5"/>'
           '<circle cx="46" cy="16" r="4" fill="#00FF66" fill-opacity=".25"/>',
           f'<text x="{width-14}" y="20" text-anchor="end" font-family="{FONT_STACK}" font-size="11" fill="#8b949e">tty1</text>',
           '<defs>']
    # clip rects (one per line) animate their width in discrete character steps
    for i, segs in enumerate(lines):
        n = sum(len(s) for s, _ in segs)
        start = t0 + i * (per_line + gap)
        end = start + per_line
        steps = n + 1
        times, widths = [0.0], [0.0]
        for k in range(steps):
            times.append((start + per_line * k / n) / loop if k < n else end / loop)
            widths.append(cw * k if k < n else cw * n)
        times.append(0.94); widths.append(cw * n)
        times.append(1.0);  widths.append(0.0)
        # keep strictly increasing key times
        ft = []
        for tkn in times:
            ft.append(max(tkn, (ft[-1] + 1e-6) if ft else 0.0))
        out.append(f'<clipPath id="c{i}"><rect x="{pad_x}" y="{pad_y + lh*i - fs}" height="{lh}" width="0">'
                   f'<animate attributeName="width" calcMode="discrete" dur="{loop}s" repeatCount="indefinite" '
                   f'keyTimes="{";".join(f"{v:.5f}" for v in ft)}" values="{";".join(f"{v:.1f}" for v in widths)}"/>'
                   f'</rect></clipPath>')
    out.append('</defs>')

    for i, segs in enumerate(lines):
        n = sum(len(s) for s, _ in segs)
        y = pad_y + lh * i
        tspans = "".join(f'<tspan fill="{c}">{esc(s)}</tspan>' for s, c in segs)
        out.append(f'<text x="{pad_x}" y="{y}" clip-path="url(#c{i})" xml:space="preserve" '
                   f'font-family="{FONT_STACK}" font-size="{fs}" textLength="{cw*n:.1f}" '
                   f'lengthAdjust="spacing">{tspans}</text>')

    # one block cursor that hops from line to line, then blinks on the last line
    xs, ys, ts = [], [], []
    for i, segs in enumerate(lines):
        n = sum(len(s) for s, _ in segs)
        start = t0 + i * (per_line + gap)
        for k in range(n + 1):
            ts.append(start + per_line * k / n)
            xs.append(pad_x + cw * k); ys.append(pad_y + lh * i - fs + 2)
    # discrete animations must begin at keyTime 0, so park the cursor at line 1 until typing starts
    ts = [0.0] + [min(t / loop, 0.93) for t in ts]
    xs.insert(0, xs[0]); ys.insert(0, ys[0])
    fixed = []
    for tkn in ts:
        fixed.append(max(tkn, (fixed[-1] + 1e-6) if fixed else 0.0))
    kt = ";".join(f"{v:.5f}" for v in fixed)
    out.append(f'<rect width="{cw:.0f}" height="{fs+1}" fill="#00FF66" x="{xs[0]}" y="{ys[0]}">'
               f'<animate attributeName="x" calcMode="discrete" dur="{loop}s" repeatCount="indefinite" keyTimes="{kt}" values="{";".join(f"{v:.1f}" for v in xs)}"/>'
               f'<animate attributeName="y" calcMode="discrete" dur="{loop}s" repeatCount="indefinite" keyTimes="{kt}" values="{";".join(f"{v:.1f}" for v in ys)}"/>'
               f'<animate attributeName="opacity" values="1;0;1" dur="0.9s" repeatCount="indefinite"/>'
               f'</rect>')
    out.append('</svg>')
    (ASSETS / "intro.svg").write_text("\n".join(out), encoding="utf-8")

# --------------------------------------------------------------------------
# 4) Footer SVG -- glitching "[ CONNECTION TERMINATED ]"
# --------------------------------------------------------------------------
def make_footer_svg():
    W, H = 760, 150
    txt = "[ CONNECTION TERMINATED ]"
    f = f'font-family="{FONT_STACK}" font-weight="700" font-size="30" text-anchor="middle" x="{W/2}"'
    sub = "session closed by shalabh.exe // see you in the next commit_"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Connection terminated">
<rect width="100%" height="100%" rx="10" fill="#000" stroke="#00FF66" stroke-opacity=".45"/>
<text {f} y="68" fill="#ff2a55" opacity=".0">{txt}
  <animate attributeName="opacity" values="0;0;.8;0;0;.8;0" keyTimes="0;.60;.61;.64;.82;.83;1" dur="5s" repeatCount="indefinite"/>
  <animate attributeName="x" values="380;380;372;380;380;389;380" keyTimes="0;.60;.61;.64;.82;.83;1" calcMode="discrete" dur="5s" repeatCount="indefinite"/>
</text>
<text {f} y="68" fill="#00dcff" opacity=".0">{txt}
  <animate attributeName="opacity" values="0;0;.8;0;0;.8;0" keyTimes="0;.60;.61;.64;.82;.83;1" dur="5s" repeatCount="indefinite"/>
  <animate attributeName="x" values="380;380;388;380;380;371;380" keyTimes="0;.60;.61;.64;.82;.83;1" calcMode="discrete" dur="5s" repeatCount="indefinite"/>
</text>
<text {f} y="68" fill="#00FF66">{txt}
  <animate attributeName="opacity" values="1;1;.4;1;1;.3;1;1" keyTimes="0;.6;.61;.64;.82;.83;.85;1" dur="5s" repeatCount="indefinite"/>
</text>
<text x="{W/2}" y="104" text-anchor="middle" font-family="{FONT_STACK}" font-size="13" fill="#8b949e">{sub}</text>
</svg>'''
    (ASSETS / "footer.svg").write_text(svg, encoding="utf-8")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("[1/4] ascii portrait ...")
    lines, dens = build_portrait()
    save_portrait(lines, dens)
    print("[2/4] matrix.gif ...")
    make_matrix_gif()
    print("[3/4] intro.svg ...")
    make_intro_svg()
    print("[4/4] footer.svg ...")
    make_footer_svg()
    print("done -> portrait.txt, assets/*")
