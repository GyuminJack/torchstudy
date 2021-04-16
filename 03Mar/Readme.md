### Results
------
#### What I've learned
- Attention
  - Two last Hidden states to one
- Dropout을 왠만하면 주는 구나.

#### Additional
- bmm (torch method)
  - torch.einsum도 비슷한 기능을 하는 경우라고 판단됨. 해당 내용을 추가적으로 확인해볼 필요가 있다.
- BLEU : Except special tokens
- validation 과정에 중간 파일을 생성하는 과정 구현

#### More
- Adadelta의 경우 PPL과 BLEU가 잘 학습되지 않는 것으로 보여짐. 모멘텀과 관련된 부분을 조절해봐야 할듯.