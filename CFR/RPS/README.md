# notes

strategy [0.4, 0.3, 0.3] means that agent picks Rock: 40%, Paper: 30%, Scissors: 30%

# train

```python
trainer = RPSTrainer(oppStrategy=[0.4, 0.3, 0.3], iterations=100000)
trainer.train()
```

# result

```python
opponent strategy:
[0.4, 0.3, 0.3]

my optimal strategy:
[0.000502654095904096, 0.9994856792374291, 1.1666666666666663e-05]
```
