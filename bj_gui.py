import tkinter as tk
from PIL import Image, ImageTk
import os
import random
import numpy as np
import cv2

from bj_logic import BlackjackGame
import bj_pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(SCRIPT_DIR, "bj_files", "background.png")
CARDS_PATH      = os.path.join(SCRIPT_DIR, "PNG-cards")
TEMPLATE_PATH = r"G:\Desktop\HTWG\2D\Projekt\Cards-Score\PNG-cards"
CARD_SIZE = (100, 145)  # Zielgröße der Kartenbilder

class BlackjackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blackjack")
        self.root.resizable(False, False)
        
        # Canvas für Hintergrund & Karten
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        
        # Hintergrund
        self.bg_img = Image.open(BACKGROUND_PATH).resize((800, 600))
        self.bg_tk = ImageTk.PhotoImage(self.bg_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_tk)
        
        # Buttons & Infos
        self.hit_button = tk.Button(self.root, text="Hit",   command=self.hit,   font=("Arial", 14), width=8)
        self.stand_button = tk.Button(self.root, text="Stand", command=self.stand, font=("Arial", 14), width=8)
        self.start_button = tk.Button(self.root, text="Start", command=self.start_round,
                                      font=("Arial", 16, "bold"), width=20)
        self.new_round_button = tk.Button(self.root, text="Neue Runde", command=self.start_round,
                                          font=("Arial", 16, "bold"), width=20)
        
        # Spielstatus-Text
        self.status_text = self.canvas.create_text(400, 40, text="", fill="white",
                                                   font=("Arial", 20, "bold"))
        
        # Pipeline & Templates
        self.pipeline = bj_pipeline
        self.templates = self.pipeline.load_card_templates(TEMPLATE_PATH)
        
        # Deck (Tk-Images + CV-Images) und Ziehstapel
        self.deck_imgs, self.cv_deck_imgs = self.load_card_images()
        self.draw_pile = self.cv_deck_imgs.copy()  # Liste von np.ndarray
        
        self.game = None
        
        # Score-Textfelder (versteckt bis Spielstart)
        self.dealer_score_text = self.canvas.create_text(700, 100, text="Dealer: 0",
                                                         fill="white", font=("Arial", 16),
                                                         state="hidden")
        self.player_score_text = self.canvas.create_text(700, 400, text="Player: 0",
                                                         fill="white", font=("Arial", 16),
                                                         state="hidden")
        # Rückseitenbild laden
        back = Image.open(os.path.join(CARDS_PATH, "back.jpg")).resize(CARD_SIZE)
        self.card_back_img = ImageTk.PhotoImage(back)

        # Ziehstapel-Anzeige (Anzahl) über dem Stapel
        self.draw_pile_text = self.canvas.create_text(20 + CARD_SIZE[0]//2, 200 - 20,
                                                      text=str(len(self.draw_pile)),
                                                      fill="white", font=("Arial", 14, "bold"),
                                                      state="hidden")

        self.show_start_button()


    def load_card_images(self):
        card_files = [f for f in os.listdir(CARDS_PATH) if f.endswith(".png")]
        tk_imgs, cv_imgs = [], []
        for fname in card_files:
            path = os.path.join(CARDS_PATH, fname)
            pil = Image.open(path).resize(CARD_SIZE)
            tk_imgs.append(ImageTk.PhotoImage(pil))
            cv_imgs.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        return tk_imgs, cv_imgs


    def show_start_button(self):
        self.clear_table()
        self.hide_all_buttons()
        self.canvas.itemconfig(self.status_text, text="BLACKJACK\nKlicke Start für neue Runde")
        self.start_button.place(x=250, y=250)
        # verstecke Scores + Ziehstapel-Anzeige
        for item in (self.dealer_score_text, self.player_score_text, self.draw_pile_text):
            self.canvas.itemconfig(item, state="hidden")


    def hide_all_buttons(self):
        for btn in (self.hit_button, self.stand_button,
                    self.start_button, self.new_round_button):
            btn.place_forget()


    def start_round(self):
        self.hide_all_buttons()
        self.clear_table()
        if len(self.draw_pile) < 4:
            self.canvas.itemconfig(self.status_text, text="Keine Karten mehr! Spiel neu starten.")
            return

        random.shuffle(self.draw_pile)
        deck_for_game = self.draw_pile.copy()
        self.game = BlackjackGame(deck_for_game, self.pipeline)
        self.game.deal_initial()

        # gezogene Karten aus Ziehstapel entfernen
        for c in self.game.player_hand + self.game.dealer_hand:
            for i, d in enumerate(self.draw_pile):
                if id(c) == id(d):
                    del self.draw_pile[i]
                    break

        self.canvas.itemconfig(self.status_text, text="Deine Aktion: Hit oder Stand")
        self.show_action_buttons()
        # zeige Scores + Ziehstapel-Anzeige
        for item in (self.dealer_score_text, self.player_score_text, self.draw_pile_text):
            self.canvas.itemconfig(item, state="normal")
        self.run_pipeline_and_update()


    def show_action_buttons(self):
        self.hit_button.place(x=220, y=550)
        self.stand_button.place(x=480, y=550)


    def clear_table(self):
        self.canvas.delete("card")


    def build_virtual_table_image(self):
        # Hintergrund in CV-Bild
        table = cv2.cvtColor(np.array(self.bg_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        # Spieler-Karten platzieren
        for idx, img in enumerate(self.game.player_hand):
            h, w, _ = img.shape
            x, y = 260 + idx*120, 380
            table[y:y+h, x:x+w] = img
        # Dealer-Karten platzieren
        for idx, img in enumerate(self.game.dealer_hand):
            h, w, _ = img.shape
            x, y = 260 + idx*120, 80
            table[y:y+h, x:x+w] = img
        # Ziehstapel (links)
        back_cv = cv2.cvtColor(np.array(Image.open(os.path.join(CARDS_PATH, "back.jpg"))
                                        .resize(CARD_SIZE)), cv2.COLOR_RGB2BGR)
        x_draw, y_draw = 20, 200
        h, w, _ = back_cv.shape
        table[y_draw:y_draw+h, x_draw:x_draw+w] = back_cv
        return table


    def run_pipeline_and_update(self):
        self.disable_buttons()
        table_img = self.build_virtual_table_image()
        detected = self.pipeline.pipeline_analyze_table(table_img, self.templates, debug=True)
        vis = table_img.copy()

        y_mid = vis.shape[0] // 2
        dealer_score, player_score = 0, 0
        for card in detected:
            cnt = card["contour"]
            bbox = card["bbox"]
            name = card["name"]
            score = card["score"]
            M = cv2.moments(cnt)
            cy = int(M["m01"] / M["m00"])
            color = (0,0,255) if cy < y_mid else (255,0,0)
            if cnt is not None:
                cv2.drawContours(vis, [cnt], -1, color, 4)
            if name:
                value = self.pipeline.get_card_value(name)
                if cy < y_mid:
                    dealer_score += value
                else:
                    player_score += value
                x, y, w, h = bbox
                value_str = name.split("_")[0]
                text = f"{value_str} ({score:.2f})"
                cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        # Ziehstapel-Anzahl
        x_draw, y_draw = 20, 200
        w, h = CARD_SIZE
        cv2.putText(vis, str(len(self.draw_pile)),
                    (x_draw + w//3, y_draw - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Scores
        dealer_score = self.game.calculate_hand_value(self.game.dealer_hand)
        player_score = self.game.calculate_hand_value(self.game.player_hand)
        cv2.putText(vis, f"Dealer: {dealer_score}", (600, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(vis, f"Player: {player_score}", (600, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Update Canvas-Textfelder
        self.canvas.itemconfig(self.dealer_score_text, text=f"Dealer: {dealer_score}")
        self.canvas.itemconfig(self.player_score_text, text=f"Player: {player_score}")

        # zurück zu Tkinter
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(vis_rgb))
        self.clear_table()
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img, tags="card")
        self.current_vis_img = tk_img  # Referenz behalten
        self.enable_buttons()



    def hit(self):
        if not self.game.player_done:
            alive = self.game.player_hit()
            self.run_pipeline_and_update()
            if not alive:
                self.show_new_round_button(self.game.status)
            else:
                self.canvas.itemconfig(self.status_text, text="Deine Aktion: Hit oder Stand")


    def stand(self):
        self.game.player_stand()
        self.game.dealer_turn()
        self.run_pipeline_and_update()
        self.show_new_round_button(self.game.status)

    def show_new_round_button(self, status_text=""):
        self.hide_all_buttons()
        self.canvas.itemconfig(self.status_text, text=f"{status_text}\nKlicke für neue Runde")
        self.new_round_button.place(x=250, y=250)


    def disable_buttons(self):
        for btn in (self.hit_button, self.stand_button,
                    self.start_button, self.new_round_button):
            btn.config(state="disabled")


    def enable_buttons(self):
        for btn in (self.hit_button, self.stand_button,
                    self.start_button, self.new_round_button):
            btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    BlackjackGUI(root)
    root.mainloop()
