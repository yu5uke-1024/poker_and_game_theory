# Demo

XFP_Kuhn_Poker.py

# notes

# config1

```python
iterations = 10000
lambda_num = 2
wandb_save = True
```

# train1

```python
kuhn_trainer = KuhnTrainer(train_iterations=iterations)
kuhn_trainer.train(lambda_num = lambda_num)
```

# result1

```python
avg util: -0.0555721046491536
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/63486375/167869865-68ca40d0-53c5-4458-a083-aaf6b1f8e952.png", width=640>
</p>

```
          Pass       Bet
Node
J     0.794771  0.205229
Jp    0.669583  0.330417
Jb    0.999950  0.000050
Jpb   0.999944  0.000056
Q     0.999650  0.000350
Qp    0.999850  0.000150
Qb    0.655584  0.344416
Qpb   0.453240  0.546760
K     0.382412  0.617588
Kp    0.000050  0.999950
Kb    0.000050  0.999950
Kpb   0.000104  0.999896
```
