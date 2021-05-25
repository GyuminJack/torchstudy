### Results
------
#### What I've learned
- Bi-directional + stacked RNN's power
- einsum
- Transfer Learning
  - 다운스트림 태스크에서 왜그렇게 쓰기 어려워 보였는지 이해가 됨.
    - 사전 학습 된 모델에서 "vocab", "tokenizer", "model" 을 가져오는 게 필수적임
    - 상기 vocab과 tokenizer를 통해 다운스트림 태스크의 데이터를 전처리해야함.
    - 그리고 model에서 사용할 특정 부분(ex, hidden_state..)을 떼어내는 부분들도 구현이 필요함

#### Additional
- reshape
  - A * B * C 를 AB * C 로 reshape하는 것과 BA * C로 reshape하는 건 다름.
  - 딥러닝 모델의 capability를 의심하기 보단 내 코드를 의심하자..
- layer nomalization 
- downstream task (실험중)

#### More
- batch, layer normalization
- various downstream tasks
