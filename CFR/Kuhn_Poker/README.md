# Demo

CFR_Kuhn_Poker.py

# notes

if wandb_save == True, you can save result on your wandb.

# config1

```python
algorithm_candicates =["vanilla_CFR", "chance_sampling_CFR", "external_sampling_MCCFR", "outcome_sampling_MCCFR"]
algo =algorithm_candicates[1]
train_iterations=10**5
num_players= 2
wandb_save = True
```

# train1

```python
kuhn_trainer = KuhnTrainer(train_iterations=train_iterations, num_players=num_players)
kuhn_trainer.train(algo)
```

# result1

```python
avg util: -0.05559434009541547

         Pass       Bet
Node
J     0.862185  0.137815
Jb    0.999985  0.000015
Jp    0.674837  0.325163
Jpb   0.999991  0.000009
K     0.593180  0.406820
Kb    0.000015  0.999985
Kp    0.000015  0.999985
Kpb   0.000013  0.999987
Q     0.999620  0.000380
Qb    0.656695  0.343305
Qp    0.999856  0.000144
Qpb   0.525814  0.474186
```

# train2 (calculate random strategy_profile exploitability)

```python
for i in range(2,6):
  kuhn_poker_agent = KuhnTrainer(train_iterations=0, num_players=i)
  print("{}player game:".format(i), kuhn_poker_agent.get_exploitability_dfs())
```

# result2

```python
2player game: 0.9166666666666665
3player game: 2.0625
4player game: 3.476041666666668
5player game: 5.01080729166667
```
