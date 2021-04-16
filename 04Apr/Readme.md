### Results
------
#### What I've learned
- Understand Transformer Architecture
  - Query, Key, Value made by linear projection
  - LM model should learn "Long Term Dependecy"
- Masking Strategy
- Contiguous (C, Fotran)
- Powerful Performance.... (BLEU > 50... in Mulit30K, only training 1epoch/10s)
- Dot-product attention divided by root(d_k) (Simple Answer for Softmax overflow)

#### Additional
- Learning Rate Scheduler
- Linear Layer to Convolution Layer

#### More
- Matrix Multiplication (einsum)
- How to Transfer Learninig