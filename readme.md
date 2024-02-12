The file _requirements.txt_ should contain all the necessary packages to run the code. I only included what was necessary for this part.

The main script is _pu_bert_trainer.py_

```
python3 pu_bert_trainer.py
```

The hyperparmeters are set in the _param_grid_ dictionary at line 53 in _pu_bert_trainer.py_

```
param_grid = {
    'learning_rate': [1e-4],
    'batch_size': [16], # unlabelled (combined) samples
    'num_epochs': [12],
    'gamma': [1.0],
    'alpha': [1.0],
    'batch_size_small': [4], # positive samples, 1 batch for each label
    'dropout': [0.5],
    'weight_decay': [0.01]
}
```

