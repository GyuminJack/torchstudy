from model import *
from data import *
import logging

class Trainer:
    def __init__(self, configs):
        input_size = configs["input_dim"]
        hidden_dim = configs["d_model"]
        output_size = configs["output_dim"]
        n_layers = configs["n_layers"]
        n_heads = configs["n_heads"]
        pf_mid_dim = configs["pf_dim"]
        dropout_rate = configs["dropout"]
        self.device = configs["device"]

        assert hidden_dim % n_heads == 0, "n_head mis-matched"

        self.model = Encoder(input_size, hidden_dim, output_size, n_layers, n_heads, pf_mid_dim, dropout_rate)
        self.model.to(self.device)

        self.set_optimizer(schedule=True)
        self.set_loss()

    def set_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def set_optimizer(self, schedule = False):
        self.optimizer = configs['optimizer']
        if schedule:
            pass
        else:
            self.scheduler = None
            self.optimizer = self.optimizer(self.model.parameters())

    def train(self, iterator):
        for batch_dict in iterator:
            self.optimizer.zero_grad()

            input_tokens = batch_dict["masked_inputs"].to(self.device)
            segments = batch_dict["segment_embedding"].to(self.device)
            attention_masks = batch_dict["attention_masks"].unsqueeze(1).unsqueeze(2).to(self.device)
            true_inputs = batch_dict["labels"].to(self.device)
            nsp_labels = batch_dict["nsp_labels"].to(self.device)
            mask_marking = batch_dict["mask_marking"].to(self.device)
            indices = (mask_marking.reshape(-1) == 1).nonzero().reshape(-1).to(self.device)

            mlm_true = torch.index_select(true_inputs.reshape(-1), 0, indices)
            mlm_labels = torch.index_select(input_tokens.reshape(-1), 0, indices)

            nsp_output, output = self.model(input_tokens, attention_masks, segments)
            mlm_output = torch.index_select(output.reshape(-1, output.size(-1)), 0, indices)

            # flatten_output = output.reshape(-1, output.size(-1))

            # Cross Entropy : NSP (nsp_labels, nsp_output)
            # Cross Entropy : MLM (mlm_labels, mlm_output)
            nsp_loss = self.loss(nsp_output, nsp_labels)
            nsp_loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            logging.debug(f"# of Total Mask token : {len(indices)}")
            logging.debug(f"Total Output Size (B, Seq, Out_vocab) : {output.size()}")
            logging.debug(f"Selected MLM output (# of Mask, Out_vocab) : {mlm_output.size()}")
            logging.debug(f"NSP output (B, 2) : {nsp_output.size()}")

        return

    def valid(self, iterator):
        pass

    def run(self, train_iter, valid_iter, test_iter):
        self.train(train_iter)


if __name__ == "__main__":
    import sys

    try:
        mode = sys.argv[1]
        if mode == "debug":
            logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    except:
        pass

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/namu_2021060809"
    txt_path = "/home/jack/torchstudy/05May/ELMo/data/ynat/train_tokenized.ynat"

    dataset = KoDataset_nsp_mlm(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(dataset, collate_fn=lambda batch: dataset.collate_fn(batch), batch_size=256)

    config = {
        "input_dim": dataset.tokenizer.vocab_size,
        "d_model": 768,
        "output_dim": dataset.tokenizer.vocab_size,
        "n_layers": 12,
        "n_heads": 12,
        "pf_dim": 1024,
        "dropout": 0.1,
        "device": "cuda:1",
        "optimizer" : torch.optim.Adam
    }
    trainer = Trainer(config)

    trainer.run(train_data_loader, None, None)
