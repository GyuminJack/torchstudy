
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
- KLUE-NER

#### More
- various downstream tasks
- GPT
- Huggingface pretrained model