
### Results
------
#### What I've learned
- BERT Architecture + NER
  - BERT is real-bidirectional Model. (Elmo is shallow. forward, backword are not directly connected)
- 느낌상..
  - 데이터가 GB단위가 아닐 경우에는 작은 배치 사이즈 (~64), 작은 LR (1e-6?)정도에서 피팅이 되는지 확인해야함
  - nsp loss는 1보다 작으며, mlm loss 학습마다 작아지지만 ~5까지는 작아져야 하는것으로 보임. 
  - train nsp accuracy는 무조건 90%가 넘어야 함.
- 스케쥴러는 필수적임. 로그를 찍어보고 best valid 모델 저장 후 거기서 다시 작아진 lr로 학습해야함.
- pretrained bert를 사용할 때 필수적인 것은 (1) Tokenizer (2) Model 임
- NER의 경우 CRF 레이어를 사용해 loss를 만들어 내는 것이 일반적이며, MEMM 등의 다른 모델들도 존재함.

#### Additional
- KLUE-TC
  - 엘모의 최고 f1-score가 79.1% 이지만 bert의 경우 80%정도로 약소하지만 차이가 있었다.
  - 하지만 엘모 튜닝의 경우 버트보다는 복잡해서 차라리 버트를 구성하고 파인튜닝하는게 더 좋다고 판단된다.

- KLUE-NER
  - pytorch-crf를 사용하는 모델을 구성
  - 버트의 출력값을 LSTM에 넣어주고 다시 CRF에 넣으면 1에폭에서 그렇지 않을때보다 f1 score 약 8% 성능 (72% -> 80%) 향상..
  - 버트의 로스가 낮은 모델을 사용하면 좋을것이라 생각했는데, 적당히 학습된 것을 사용하는게 더 성능이 좋았음. 
    - 버트 학습 중 만들어진 몇개의 모델을 바꿔가면서 최고의 성능을 내는 모델을 찾아야 할 것으로 판단됨.

#### More
- various downstream tasks
- GPT
- Huggingface pretrained model