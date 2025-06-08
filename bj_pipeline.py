import cv2
import numpy as np
import glob
import os
import random
from PIL import Image

TEMPLATE_PATH = r"G:\Desktop\HTWG\2D\Projekt\Cards-Score\PNG-cards"
TEMPLATE_HEIGHT, TEMPLATE_WIDTH = 500, 726  # <--- Passe an deine Kartenvorlagen an!
width = TEMPLATE_WIDTH
height = TEMPLATE_HEIGHT

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def order_points(pts):
    """
    Sortiere vier Punkte: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_leftmost_x(cnt):
    """
    Kleinste x-Koordinate einer Kontur (für Links→Rechts-Sortierung).
    """
    pts = cnt.reshape(4, 2)
    return np.min(pts[:, 0])

def extract_rank_corner(card_img, tw, th, rel_w=0.20, rel_h=0.20):
    h, w = card_img.shape[:2]
    cx, cy = int(w * rel_w), int(h * rel_h)
    corner = card_img[0:cy, 0:cx]
    corner_resized = cv2.resize(corner, (tw, th))
    return corner_resized



# ── Karten-Template-Laden ────────────────────────────────────────────────────

def load_card_templates(folder="PNG-cards"):
    templates = {}
    paths = glob.glob(os.path.join(folder, "*.png"))
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            templates[os.path.basename(path)] = img
    return templates


# ── Bildverschlechterung für Realismus ──────────────────────────────────────

def degrade_card(img, blur_kernel=(5, 5), noise_std=5):
    """
    Macht eine Karte 'unkenntlicher', indem Blur und Rauschen hinzugefügt werden.
    """
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    noise = np.random.normal(0, noise_std, blurred.shape).astype(np.uint8)
    degraded = cv2.add(blurred, noise)
    return degraded

# ── Karten auf virtuellem Spielfeld platzieren ──────────────────────────────

def create_random_blackjack_table(card_imgs, num_player=2, num_dealer=2, img_size=(600, 300)):
    """
    Legt eine zufällige Auswahl an Kartenbildern für Spieler und Dealer nebeneinander auf einem leeren Bild ab.
    Gibt das zusammengesetzte Spielfeld-Bild und die verwendeten Karten zurück.
    """
    # Shuffle und Auswahl
    deck = card_imgs.copy()
    random.shuffle(deck)
    selected = deck[:num_player + num_dealer]
    player_cards = selected[:num_player]
    dealer_cards = selected[num_player:num_player+num_dealer]

    table_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 30  # dunkler Hintergrund

    # Karten nebeneinander platzieren
    offset_x = 40
    for idx, card in enumerate(player_cards):
        h, w, _ = card.shape
        table_img[30:30+h, offset_x:offset_x+w] = card
        offset_x += w + 20

    offset_x = 40
    for idx, card in enumerate(dealer_cards):
        h, w, _ = card.shape
        table_img[150:150+h, offset_x:offset_x+w] = card
        offset_x += w + 20

    return table_img, player_cards, dealer_cards

# ── Pipeline-Schritt: Kartenerkennung ───────────────────────────────────────

def find_cards(img, min_area=5000):
    """
    Findet Karten im Bild anhand der Konturen (rechteckig).
    Gibt eine Liste der extrahierten Kartenbilder zurück.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 60, 120)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_imgs = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(table_img, M, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))
            card_imgs.append(warp)
    return card_imgs

# ── Pipeline-Schritt: Template-Matching ─────────────────────────────────────

def preprocess(img, target_size):
    img = cv2.resize(img, target_size)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

    


def match_card(card_img, templates, debug=False, min_score=0.6):
    best_score, best_name = None, None
    scores_dict = {}

    for i, (name, tmpl) in enumerate(templates.items()):
        th, tw = tmpl.shape[:2]
        card_resized = cv2.resize(card_img, (tw, th))
        tmpl_resized = cv2.resize(tmpl, (tw, th))
        card_gray = cv2.cvtColor(card_resized, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(card_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        score = float(res.max())
        scores_dict[name] = score
        if best_score is None or score > best_score:
            best_score = score
            best_name = name

    if debug:
        print("[DEBUG] --- Top 3 Matching Scores ---")
        top3 = sorted(scores_dict.items(), key=lambda x: -x[1])[:3]
        for k, v in top3:
            print(f"  {k}: {v:.3f}")
        if best_score is not None:
            print(f"[DEBUG] BEST MATCH: {best_name} (score={best_score:.3f})")
        else:
            print(f"[DEBUG] BEST MATCH: None (score=None)")
    if best_score is None or best_score < min_score:
        return None, None
    return best_name, best_score









# ── Pipeline-Komplettausführung ─────────────────────────────────────────────

def pipeline_analyze_table(table_img, templates, return_contours=True, debug=False):
    """
    Findet Karten im Tischbild, extrahiert sie (perspektivisch entzerrt)
    und matched JEDE GANZE Karte mit den PNG-cards-Templates.
    """
    detected_cards = []

    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 60, 120)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Nutze die Größe der ersten PNG-Template-Karte als Zielgröße für ALLE Warps
    sample_template = next(iter(templates.values()))
    th, tw = sample_template.shape[:2]

    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 5000:
            pts = approx.reshape(4,2)
            rect = order_points(pts)
            dst = np.array([
                [0, 0],
                [tw - 1, 0],
                [tw - 1, th - 1],
                [0, th - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(table_img, M, (tw, th))
            if debug:
                print(f"[DEBUG] Warp {i} shape: {warp.shape}")
            name, score = match_card(warp, templates, debug=debug, min_score=0.6)
            if debug:
                print(f"[DEBUG] Matching result for card {i}: {name} (score={score})")
            bbox = cv2.boundingRect(cnt)
            detected_cards.append({
                "name":    name,
                "score":   score,
                "img":     warp,
                "contour": cnt if return_contours else None,
                "bbox":    bbox,
            })
    return detected_cards









# ── Hilfsfunktion: Deck laden ───────────────────────────────────────────────

def load_deck_images(folder="PNG-cards"):
    """
    Lädt alle Kartenbilder als Liste aus dem gegebenen Ordner.
    """
    paths = glob.glob(os.path.join(folder, "*.png"))
    cards = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            cards.append(img)
    return cards

# ── (Optional) Kartenscore extrahieren ──────────────────────────────────────

def extract_rank_from_filename(filename):
    """
    Extrahiert den Kartenwert aus dem Dateinamen.
    Beispiel: "card_7H.png" → "7H"
    """
    base = os.path.splitext(filename)[0]
    return base.split("_")[-1]

# ── (Optional) Blackjack-Kartenwert extrahieren ───────────────────────────

def get_card_value(card_name):
    """
    Gibt den Blackjack-Wert basierend auf Dateinamen wie 'ace_of_spades.png'.
    Ass = 1, Bube/Dame/König = 10, Zahlenkarten = int.
    """
    rank = card_name.split("_")[0].lower()
    if rank == "ace":
        return 1
    if rank in ["jack", "queen", "king"]:
        return 10
    try:
        return int(rank)
    except ValueError:
        return 0


# ── __main__ für Tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Beispieltest: Laden und Zusammenbauen eines Blackjack-Tisches
    deck_imgs = load_deck_images("PNG-cards")
    card_templates = load_card_templates(TEMPLATE_PATH)
    # Karten künstlich verschlechtern:
    deck_imgs = [degrade_card(img) for img in deck_imgs]
    table_img, player_cards, dealer_cards = create_random_blackjack_table(deck_imgs)

    # Pipeline testen
    detected = pipeline_analyze_table(table_img, card_templates)
    for d in detected:
        print(f"Karte erkannt: {d['name']} (Score: {d['score']:.2f})")

    # (Optional: Anzeige mit cv2.imshow hier)
