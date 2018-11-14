# min-char-rnn
A RNN model for character prediction, built on the basis of the [code](https://gist.github.com/karpathy/d4dee566867f8291f086) written by [Andrej Karpathy](https://karpathy.github.io/).

### Requirements: 
- Python 3.5  
- Numpy  

### Description:
`model`: Parameters obtained by training process.  
`input.txt`: A sequence of characters arranged in a particular order, serving as training dataset.  
`min-char-rnn.py`: Entry point for this project.  

`TRAIN_DATA.npy`: Meta-data created during training process (epoch, iteration, loss).

### Training:
Uncomment and run `model = Min_char_rnn()` and `model.train()` in `min-char-rnn.py`. The updated weights and parameters will be saved in `'model'` folder.

### Testing:
Uncomment and run `model = Min_char_rnn(1)` and `model.test()` in `min-char-rnn.py` to predict a sequence of characters similar to training dataset.  
