import numpy as np

card_order =  {'J':0, 'Q':1, 'K':2}
cards = ['J', 'Q', 'K']
ACTION_DICT_verse = {"f":0, "c":1, "r":2}

def make_state_bit(X):
 X_bit = np.array([0 for _ in range(30)])

 if X == None:
   return X_bit
 X_bit[card_order[X[0]]] = 1

 for idx, Xi in enumerate(X[1:]):
   if Xi not in cards:
     X_bit[(2+1) + 3*idx + ACTION_DICT_verse[Xi]] = 1
   else:
     com_idx = card_order[Xi] + 1
     X_bit[-com_idx] = 1
 return X_bit

print(make_state_bit("JrcKc"))
print(make_state_bit("JrcK"))
