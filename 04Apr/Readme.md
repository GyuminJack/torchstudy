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
- Xavier Initialize is very Useful (Bleu 30 to 50)

#### Additional
- Learning Rate Scheduler
- Linear Layer to Convolution Layer
- KoKoBibleDataset
  - 처음에는 잘 안됨.. (loss가 5밑으로 안떨어짐..) 
  - 문제가 LSTM seq2seq에서 사용한 vocab을 가져왔는데 거기서 역순으로 데이터를 출력하고 있었음. 
  - 이걸 바꾸니 loss 가 쭉쭉 떨어짐 :) -> 근데 또 언제는 떨어지고, 언제는 안떨어지고.. 
    - LR 스케줄러 썼더니.. 엄청 잘됨.... 너무 놀라ㅇ움...
  
#### More
- Matrix Multiplication (einsum)
- How to Transfer Learninig

