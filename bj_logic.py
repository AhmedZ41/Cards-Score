import random

TEMPLATE_PATH = r"G:\Desktop\HTWG\2D\Projekt\Cards-Score\PNG-cards"

class BlackjackGame:
    def __init__(self, deck, pipeline):
        self.deck = deck.copy()
        random.shuffle(self.deck)
        self.pipeline = pipeline
        self.player_hand = []
        self.dealer_hand = []
        self.player_done = False
        self.bet = 0
        self.dealer_hide = True
        self.status = ""
        self.result = None

    def set_bet(self, amount):
        self.bet = amount

    def deal_initial(self):
        # Initialdeals laut Regel
        self.player_hand = [self.draw_card()]
        self.dealer_hand = [self.draw_card()]  # verdeckt
        self.player_hand.append(self.draw_card())
        self.dealer_hand.append(self.draw_card())  # offen


    def draw_card(self):
        if not self.deck:
            raise Exception("Deck ist leer!")
        return self.deck.pop()


    def player_hit(self):
        if not self.player_done:
            self.player_hand.append(self.draw_card())
            if self.calculate_hand_value(self.player_hand) > 21:
                self.status = "Überkauft! Dealer gewinnt."
                self.result = "lose"
                self.player_done = True
                return False
        return True

    def player_stand(self):
        self.player_done = True

    def reveal_dealer_card(self):
        self.dealer_hide = False

    def dealer_turn(self):
        self.reveal_dealer_card()
        while self.calculate_hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
            if self.calculate_hand_value(self.dealer_hand) > 21:
                self.status = "Dealer überkauft! Spieler gewinnt."
                self.result = "win"
                return
        self.determine_winner()

    def determine_winner(self):
        player = self.calculate_hand_value(self.player_hand)
        dealer = self.calculate_hand_value(self.dealer_hand)
        if player > 21:
            self.status = "Überkauft! Dealer gewinnt."
            self.result = "lose"
        elif dealer > 21:
            self.status = "Dealer überkauft! Spieler gewinnt."
            self.result = "win"
        elif player > dealer:
            self.status = "Spieler gewinnt!"
            self.result = "win"
        elif player < dealer:
            self.status = "Dealer gewinnt!"
            self.result = "lose"
        else:
            self.status = "Unentschieden!"
            self.result = "draw"

    def calculate_hand_value(self, hand):
        total = 0
        aces = 0
        templates = self.pipeline.load_card_templates(TEMPLATE_PATH)
        for idx, card_img in enumerate(hand):
            name, score = self.pipeline.match_card(card_img, templates, debug=False)  # <-- debug AUS!
            if name is None:
                print(f"[DEBUG] No match for hand card {idx}!")
                value = 0
            else:
                value = self.pipeline.get_card_value(name)
                print(f"[DEBUG] Matched hand card {idx}: {name} → Value: {value}")

            if value == 1:
                aces += 1
            total += value
        while aces and total + 10 <= 21:
            total += 10
            aces -= 1
        print(f"[DEBUG] Hand-Value calculated: {total}")
        return total




    def can_split(self):
        # Optional: Implementiere Split
        return False

    def reset(self):
        self.__init__(self.deck, self.pipeline)
