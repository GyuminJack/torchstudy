### Results
------
#### What I've learned
- seq2seq Base Model
- Teacher Forcing Technic
- RNN Batch is [seq_len * batch]
- Rervese Sentence Get good Performance

#### Additional
- Torch Implement of RNN modules
- Train with File I/O (Slow but Useful)
- padding & packing
  - padding : Equalize All of Tokens in Minibatch
  - packing : To utilize Computation, Tear Batches
- default PAD_IDX = 0


#### KO-EN test
- Not Preprocessing. only Tokenize with Khaiii
  - English words in Korean vocabs.
  - No good vocabs quality, so Train is not well doing.
- When I put "미국" in Source, sometimes translated word is 'US', other 'United States'.
- It is very interested

#### More
- RNN + Attention
- Permute 하는 과정에 대한 이해