
import itertools
from re import A
import numpy as np

num_players = 2
def card_distribution(num_players):
  card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
  card_deck = []
  for i in range(num_players+1):
    card_deck.append(card[11-num_players+i])
    card_deck.append(card[11-num_players+i])

  return card_deck

cards = card_distribution(num_players)
print(cards)
cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, 3)]
print(len(cards_candicates))
