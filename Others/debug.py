import itertools
import collections


num_player = 2
def card_distribution():
  card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
  card_deck = []
  for i in range(num_player+1):
    card_deck.append(card[11-num_player+i])
    card_deck.append(card[11-num_player+i])

  return card_deck

cards = card_distribution()

a = collections.Counter([cards_candicate for cards_candicate in itertools.permutations(cards, num_player+1)])
print(a)
