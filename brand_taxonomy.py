# ============================================================
# brand_taxonomy.py
# Rule-based brand taxonomy for competitor classification
#
# KNOWN_BRANDS maps any brand/product keyword (lowercase) to a
# canonical brand name. Used by classify_mentions() in Cell 7 to
# determine whether a mentioned brand is a cross-brand competitor
# or an in-brand (cannibalization) reference.
#
# Focal brands in dataset: Apple, Samsung, Google, LG, Motorola, Huawei
# All other recognised brands are treated as cross-brand competitors
# when mentioned alongside a focal product.
# ============================================================

KNOWN_BRANDS = {
    # ── Apple ─────────────────────────────────────────────
    "apple":          "apple",
    "iphone":         "apple",
    "ipad":           "apple",
    "ios":            "apple",
    "macbook":        "apple",
    "airpods":        "apple",

    # ── Samsung ───────────────────────────────────────────
    "samsung":        "samsung",
    "galaxy":         "samsung",
    "exynos":         "samsung",
    "galaxy s":       "samsung",
    "galaxy a":       "samsung",
    "galaxy j":       "samsung",
    "galaxy note":    "samsung",
    "galaxy z":       "samsung",

    # ── Google ────────────────────────────────────────────
    "google":         "google",
    "pixel":          "google",
    "nexus":          "google",

    # ── LG ────────────────────────────────────────────────
    "lg":             "lg",
    "lg g":           "lg",
    "lg v":           "lg",
    "lg stylo":       "lg",
    "lg velvet":      "lg",
    "lg wing":        "lg",

    # ── Motorola ──────────────────────────────────────────
    "motorola":       "motorola",
    "moto":           "motorola",
    "moto g":         "motorola",
    "moto e":         "motorola",
    "moto z":         "motorola",
    "razr":           "motorola",

    # ── Huawei / Honor ────────────────────────────────────
    "huawei":         "huawei",
    "honor":          "huawei",
    "kirin":          "huawei",
    "mate":           "huawei",   # Huawei Mate series

    # ── OnePlus ───────────────────────────────────────────
    "oneplus":        "oneplus",
    "one plus":       "oneplus",
    "warp charge":    "oneplus",
    "oxygenos":       "oneplus",

    # ── Sony ──────────────────────────────────────────────
    "sony":           "sony",
    "xperia":         "sony",

    # ── Nokia ─────────────────────────────────────────────
    "nokia":          "nokia",

    # ── HTC ───────────────────────────────────────────────
    "htc":            "htc",

    # ── Xiaomi / Redmi / POCO ─────────────────────────────
    "xiaomi":         "xiaomi",
    "redmi":          "xiaomi",
    "poco":           "xiaomi",
    "miui":           "xiaomi",

    # ── Oppo ──────────────────────────────────────────────
    "oppo":           "oppo",
    "coloros":        "oppo",
    "find x":         "oppo",
    "reno":           "oppo",

    # ── Vivo ──────────────────────────────────────────────
    "vivo":           "vivo",
    "funtouch":       "vivo",

    # ── Realme ────────────────────────────────────────────
    "realme":         "realme",

    # ── Nothing ───────────────────────────────────────────
    "nothing phone":  "nothing",
    "nothing":        "nothing",

    # ── BlackBerry ────────────────────────────────────────
    "blackberry":     "blackberry",
    "bb10":           "blackberry",

    # ── Microsoft ─────────────────────────────────────────
    "microsoft":      "microsoft",
    "surface":        "microsoft",
    "windows phone":  "microsoft",
    "lumia":          "microsoft",

    # ── Asus ──────────────────────────────────────────────
    "asus":           "asus",
    "zenfone":        "asus",
    "rog phone":      "asus",

    # ── ZTE ───────────────────────────────────────────────
    "zte":            "zte",
    "blade":          "zte",

    # ── TCL ───────────────────────────────────────────────
    "tcl":            "tcl",
    "alcatel":        "tcl",   # Alcatel is a TCL brand

    # ── Fairphone ─────────────────────────────────────────
    "fairphone":      "fairphone",

    # ── CAT / Caterpillar ─────────────────────────────────
    "cat phone":      "cat",
    "caterpillar":    "cat",

    # ── Samsung model shorthand ───────────────────────────
    # S-series
    "galaxy s2":      "samsung", "galaxy s3":      "samsung",
    "galaxy s4":      "samsung", "galaxy s5":      "samsung",
    "galaxy s6":      "samsung", "galaxy s6 edge": "samsung",
    "galaxy s7":      "samsung", "galaxy s7 edge": "samsung",
    "galaxy s8":      "samsung", "galaxy s8+":     "samsung",
    "galaxy s9":      "samsung", "galaxy s9+":     "samsung",
    "galaxy s10":     "samsung", "galaxy s10e":    "samsung",
    "galaxy s10+":    "samsung", "galaxy s20":     "samsung",
    "galaxy s20 fe":  "samsung", "galaxy s20+":    "samsung",
    "galaxy s20 ultra": "samsung",
    "galaxy s21":     "samsung", "galaxy s21 fe":  "samsung",
    "galaxy s21+":    "samsung", "galaxy s21 ultra": "samsung",
    "galaxy s22":     "samsung", "galaxy s22+":    "samsung",
    "galaxy s22 ultra": "samsung",
    "galaxy s23":     "samsung", "galaxy s23 fe":  "samsung",
    "galaxy s23+":    "samsung", "galaxy s23 ultra": "samsung",
    # Bare S-series shorthand (e.g. "the s20", "s23 ultra", "my s9")
    # Unpadded versions for when model number appears at start/end of string
    "s20 fe":    "samsung", "s21 fe":    "samsung", "s23 fe":    "samsung",
    "s20 ultra": "samsung", "s21 ultra": "samsung",
    "s22 ultra": "samsung", "s23 ultra": "samsung",
    "s20+":  "samsung", "s21+":  "samsung", "s22+":  "samsung", "s23+":  "samsung",
    "s20":   "samsung", "s21":   "samsung", "s22":   "samsung", "s23":   "samsung",
    "s10e":  "samsung", "s10+":  "samsung", "s10":   "samsung",
    "s9+":   "samsung", "s9":    "samsung",
    "s8+":   "samsung", "s8":    "samsung",
    "s7 edge": "samsung", "s7":  "samsung",
    "s6 edge+": "samsung", "s6 edge": "samsung", "s6":  "samsung",
    "s5":    "samsung", "s4":    "samsung", "s3":    "samsung", "s2":    "samsung",
    # Note series
    "galaxy note":    "samsung",
    "note 4":  "samsung", "note 5":  "samsung", "note 7":  "samsung",
    "note 8":  "samsung", "note 9":  "samsung", "note 10": "samsung",
    "note 20": "samsung", "note 20 ultra": "samsung",
    " note4 ": "samsung", " note5 ": "samsung", " note7 ": "samsung",
    " note8 ": "samsung", " note9 ": "samsung",
    # Z-series foldables
    "galaxy z fold":  "samsung", "galaxy z flip":  "samsung",
    "z fold":   "samsung", "z fold 2": "samsung", "z fold 3": "samsung",
    "z fold 4": "samsung", "z fold 5": "samsung",
    "z flip":   "samsung", "z flip 3": "samsung", "z flip 4": "samsung",
    "z flip 5": "samsung",
    # A-series shorthand
    "galaxy a": "samsung",
    "a10": "samsung", "a11": "samsung", "a12": "samsung",
    "a21": "samsung", "a32": "samsung", "a51": "samsung",
    "a52": "samsung", "a53": "samsung",
    # J-series shorthand
    "galaxy j": "samsung",
    "j3": "samsung", "j7": "samsung",

    # ── Apple model shorthand ─────────────────────────────
    # iPhone (iphone already maps to apple, but model numbers help disambiguation)
    "iphone 6":       "apple", "iphone 6 plus":  "apple",
    "iphone 6s":      "apple", "iphone 6s plus": "apple",
    "iphone 7":       "apple", "iphone 7 plus":  "apple",
    "iphone 8":       "apple", "iphone 8 plus":  "apple",
    "iphone x":       "apple", "iphone xr":      "apple",
    "iphone xs":      "apple", "iphone xs max":  "apple",
    "iphone 11":      "apple", "iphone 11 pro":  "apple",
    "iphone 11 pro max": "apple",
    "iphone 12":      "apple", "iphone 12 mini": "apple",
    "iphone 12 pro":  "apple", "iphone 12 pro max": "apple",
    "iphone 13":      "apple", "iphone 13 mini": "apple",
    "iphone 13 pro":  "apple", "iphone 13 pro max": "apple",
    "iphone 14":      "apple", "iphone 14 plus": "apple",
    "iphone 14 pro":  "apple", "iphone 14 pro max": "apple",
    "iphone 15":      "apple", "iphone 15 plus": "apple",
    "iphone 15 pro":  "apple", "iphone 15 pro max": "apple",
    "iphone se":      "apple",
    # iPad (tablets — cross-brand competitor if comparing phone to iPad, but included)
    "ipad":           "apple",
    "ipad pro":       "apple", "ipad air":       "apple",
    "ipad mini":      "apple", "ipad 9th":       "apple",
    "ipad 10th":      "apple",
    # Mac (included for completeness — rarely a direct phone competitor)
    "macbook pro":    "apple", "macbook air":    "apple",
    "macbook":        "apple", "mac mini":       "apple",
    "imac":           "apple", "mac pro":        "apple",
    "macos":          "apple",

    # ── Google model shorthand ────────────────────────────
    "pixel 1":        "google", "pixel 2":        "google",
    "pixel 2 xl":     "google", "pixel 3":        "google",
    "pixel 3 xl":     "google", "pixel 3a":       "google",
    "pixel 3a xl":    "google", "pixel 4":        "google",
    "pixel 4 xl":     "google", "pixel 4a":       "google",
    "pixel 4a 5g":    "google", "pixel 5":        "google",
    "pixel 5a":       "google", "pixel 6":        "google",
    "pixel 6a":       "google", "pixel 6 pro":    "google",
    "pixel 7":        "google", "pixel 7 pro":    "google",
    "pixel 7a":       "google", "pixel 8":        "google",
    "pixel 8 pro":    "google", "pixel fold":     "google",
    # Google Nexus series
    "nexus 4":        "google", "nexus 5":        "google",
    "nexus 5x":       "google", "nexus 6":        "google",
    "nexus 6p":       "google", "nexus 7":        "google",
    "nexus 9":        "google", "nexus 10":       "google",

    # ── Huawei model shorthand ────────────────────────────
    "p8":   "huawei", "p9":   "huawei", "p10":  "huawei",
    "p20":  "huawei", "p30":  "huawei", "p40":  "huawei",
    "p50":  "huawei", "p60":  "huawei",
    "p20 pro": "huawei", "p30 pro": "huawei",
    "p40 pro": "huawei", "p50 pro": "huawei",
    "mate 8":  "huawei", "mate 9":  "huawei", "mate 10": "huawei",
    "mate 20": "huawei", "mate 30": "huawei", "mate 40": "huawei",
    "mate 20 pro": "huawei", "mate 30 pro": "huawei",

    # ── LG model shorthand ────────────────────────────────
    "lg g2":   "lg", "lg g3":   "lg", "lg g4":   "lg",
    "lg g5":   "lg", "lg g6":   "lg", "lg g7":   "lg",
    "lg g8":   "lg", "lg v30":  "lg", "lg v40":  "lg",
    "lg v50":  "lg", "lg v60":  "lg",
    "lg velvet": "lg", "lg wing": "lg",
    "stylo 4": "lg", "stylo 5": "lg", "stylo 6": "lg",

    # ── Motorola model shorthand ──────────────────────────
    "moto g4": "motorola", "moto g5": "motorola", "moto g6": "motorola",
    "moto g7": "motorola", "moto g8": "motorola", "moto g9": "motorola",
    "moto g fast": "motorola", "moto g power": "motorola",
    "moto g play":  "motorola", "moto g stylus": "motorola",
    "moto z2":  "motorola", "moto z3":  "motorola", "moto z4":  "motorola",
    "moto edge": "motorola", "moto edge+": "motorola",
    "razr 2019": "motorola", "razr 2020": "motorola", "razr 5g": "motorola",

    # ── OnePlus model shorthand ───────────────────────────
    "oneplus 3":  "oneplus", "oneplus 5":  "oneplus",
    "oneplus 6":  "oneplus", "oneplus 7":  "oneplus",
    "oneplus 8":  "oneplus", "oneplus 9":  "oneplus",
    "oneplus 10": "oneplus", "oneplus 11": "oneplus",
    "oneplus nord": "oneplus",
    "op3": "oneplus", "op5": "oneplus", "op6": "oneplus",
    "op7": "oneplus", "op8": "oneplus",

    # ── Sony model shorthand ──────────────────────────────
    "xperia 1":   "sony", "xperia 5":   "sony",
    "xperia 10":  "sony", "xperia xz":  "sony",
    "xperia z":   "sony",

    # ── Nokia model shorthand ─────────────────────────────
    "nokia 6":  "nokia", "nokia 7":  "nokia", "nokia 8":  "nokia",
    "nokia 9":  "nokia", "nokia 3.4": "nokia", "nokia 5.4": "nokia",
}

# Keywords that identify each focal brand in extracted mentions
# Used to check if an extracted brand name refers to the focal brand itself
FOCAL_BRAND_KEYS = {
    "apple":    {"apple", "iphone", "ios", "ipad", "macbook", "imac", "macos"},
    "samsung":  {"samsung", "galaxy", "exynos", " s2 ", " s3 ", " s4 ", " s5 ",
                 " s6 ", " s7 ", " s8 ", " s9 ", " s10 ", " s20 ", " s21 ",
                 " s22 ", " s23 ", "note 4", "note 5", "note 7", "note 8",
                 "note 9", "note 10", "note 20", "z fold", "z flip"},
    "google":   {"google", "pixel", "nexus"},
    "lg":       {"lg", "stylo", "velvet", "wing"},
    "motorola": {"motorola", "moto", "razr"},
    "huawei":   {"huawei", "honor", "kirin", "mate", "nova"},
}