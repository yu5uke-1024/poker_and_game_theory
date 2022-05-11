# Demo

FP_Game_RPS.py

# notes

# config1 - coin game -

```python
me_strategy = np.array([0.1, 0.9], dtype = "float")
opp_strategy = np.array([0.6, 0.4], dtype = "float")
iterations = 10000
payoff = [[1,-1], [-1, 1], [-1,1], [1, -1]]
```

# train1

```python
trainer1 = GameTrainer(iterations = iterations, avg_me_strategy=me_strategy, avg_opp_strategy=opp_strategy, payoff=payoff)
result1 = trainer1.train()
print("player0 strategy:", result1[0], "player1 strategy:", result1[1])
```

# result1

```python
player0 strategy: [0.49946005 0.50053995] player1 strategy: [0.50650935 0.49349065]
```

# config2 - RPS game -

```python
me_strategy = np.array([0.4, 0.4, 0.2], dtype = "float")
opp_strategy = np.array([0.0, 0.3, 0.7], dtype = "float")
iterations = 1000
```

# train2

```python
trainer2 = RPSTrainer(iterations = iterations, avg_me_strategy=me_strategy, avg_opp_strategy=opp_strategy)
result2  = trainer2.train()
print("player0 strategy:", result2[0], "player1 strategy:", result2[1])
```

# result2

```python
player0 strategy: [0.34105894 0.32007992 0.33886114] player1 strategy: [0.31768232 0.35194805 0.33036963]
```
