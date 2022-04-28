
import itertools
import numpy as np

num_players = 2


def card_distribution():
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    card_deck = []
    for i in range(num_players+1):
      card_deck.append(card[11-num_players+i])
      card_deck.append(card[11-num_players+i])

    return card_deck

def make_rank():
  card_deck = card_distribution()
  card_unique = card_deck[::num_players]
  card_rank = {}
  count = (len(card_unique)-1)*len(card_unique) //2
  for i in range(len(card_unique)-1,-1, -1):
    for j in range(i-1, -1, -1):
          card_rank[card_unique[i] + card_unique[j]] = count
          card_rank[card_unique[j] + card_unique[i]] = count
          count -= 1

  count = (len(card_unique)-1)*len(card_unique) //2 +1
  for i in range(len(card_unique)):
      card_rank[card_unique[i] + card_unique[i]] = count
      count += 1
  return card_rank

card_rank = make_rank()

def Rank( my_card, com_card):
  hand = my_card + com_card
  return card_rank[hand]


def Split_history(history):
  for ai, history_ai in enumerate(history[num_players:]):
    if history_ai in card_distribution():
      idx = ai+num_players
      community_catd = history_ai
  return history[:num_players], history[num_players:idx], community_catd, history[idx+1:]


def action_history_player(history):
  #target_player_iのaction 履歴
  player_action_list = [[] for _ in range(num_players)]
  player_money_list_round1 = [1 for _ in range(num_players)]
  player_money_list_round2 = [0 for _ in range(num_players)]

  f_count, a_count, raise_count = 0, 0, 0

  card = card_distribution()
  private_cards, history_before, community_card, history_after = Split_history(history)
  for hi in history_before:
    while len(player_action_list[(a_count + f_count)%num_players])>=1 and player_action_list[(a_count + f_count)%num_players][-1] == "f":
      f_count += 1
    player_action_list[(a_count + f_count)%num_players].append(hi)

    if hi == "c":
      player_money_list_round1[(a_count + f_count)%num_players] = max(player_money_list_round1)
    elif hi == "r" and raise_count == 0:
      raise_count += 1
      player_money_list_round1[(a_count + f_count)%num_players] += 1
    elif hi == "r" and raise_count == 1:
      player_money_list_round1[(a_count + f_count)%num_players] += 3

    a_count += 1

  f_count, a_count, raise_count = 0, 0, 0

  for hi in history_after:
      if hi not in card:
        while len(player_action_list[(a_count + f_count)%num_players])>=1 and player_action_list[(a_count + f_count)%num_players][-1] == "f":
          f_count += 1
        player_action_list[(a_count + f_count)%num_players].append(hi)

        if hi == "c":
          player_money_list_round2[(a_count + f_count)%num_players] = max(player_money_list_round2)
        elif hi == "r" and raise_count == 0:
          raise_count += 1
          player_money_list_round2[(a_count + f_count)%num_players] += 4
        elif hi == "r" and raise_count == 1:
          player_money_list_round2[(a_count + f_count)%num_players] += 8

        a_count += 1

  return player_action_list, player_money_list_round1, player_money_list_round2, community_card


def Return_payoff_for_terminal_states(history, target_player_i):
  #target_player_iのaction 履歴
  player_action_list, player_money_list_round1, player_money_list_round2, community_card = action_history_player(history)

  # target_player_i :fold
  if player_action_list[target_player_i][-1] == "f":
    return -player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

  #周りがfold
  last_play =[hi[-1] for idx, hi in enumerate(player_action_list) if idx != target_player_i]
  if last_play.count("f") == num_players - 1:
    return sum(player_money_list_round1) + sum(player_money_list_round2) - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

  #何人かでshow down
  show_down_player =[idx for idx, hi in enumerate(player_action_list) if hi[-1] != "f"]
  show_down_player_card = {}
  for idx in show_down_player:
    show_down_player_card[idx] = Rank(history[idx], community_card)
  max_rank = max(show_down_player_card.values())
  if show_down_player_card[target_player_i] != max_rank:
    return - player_money_list_round1[target_player_i] - player_money_list_round2
  else:
    win_num = len([idx for idx, card_rank in show_down_player_card.items() if card_rank == max_rank])
    return (sum(player_money_list_round1) + sum(player_money_list_round2))/win_num - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]


print(Return_payoff_for_terminal_states("KJTcccKcrcrcf", 0))


def action_player(history):
  if
  return

print(action_player("KJTcc"))
 len(infoset_without_hand_card) == 0 or infoset_without_hand_card.count("r") == 0
