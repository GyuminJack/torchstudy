
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
- LM은 생각보다 더 학습을 시켜야됨.. 적당히 된거 같아서 해보면 실제 성능을 전부 뽑아내지 못하는 결과를 가져올 수 있음
- bi-directional + stack은 거의 필수적일것이라 생각됨.
  - bi-directional을 통해서 양방향 학습이 가능하지만 실제 추론 과정에서는 한방향의 입력만 받아내기 때문에 양방향을 모두 이용하려면 stack을 해서 위 층의 rnn으로 전파해줘야 전체 맥락을 파악할 수 있음.
  
#### Additional
- reshape
  - A * B * C 를 AB * C 로 reshape하는 것과 BA * C로 reshape하는 건 다름.
  - 딥러닝 모델의 capability를 의심하기 보단 내 코드를 의심하자..
- layer nomalization 
- downstream task
<img width="993" alt="스크린샷 2021-05-25 오후 5 36 40" src="https://user-images.githubusercontent.com/32768535/119466671-d2cbcd00-bd7f-11eb-894a-3e75562f1b9b.png">

<img width="946" alt="스크린샷 2021-05-25 오후 5 36 48" src="https://user-images.githubusercontent.com/32768535/119466685-d7908100-bd7f-11eb-96d8-7bfac75e72f0.png">

#### More
- batch, layer normalization
- various downstream tasks
