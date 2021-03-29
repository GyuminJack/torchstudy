from models import *
from data import EnFrData
from trainer import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import math
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model_with_hp(config):
    dataObj = EnFrData(config) # device, batch_size
    
    config['input_dim'] = len(dataObj.SRC.vocab)
    config['output_dim'] = len(dataObj.TRG.vocab)
    config['pad_idx'] = dataObj.TRG.vocab.stoi[dataObj.TRG.pad_token] 
    trainer = Trainer(config) # input_dim, output_dim, emb_dim, hid_dim, emb_dim

    CLIP = 1
    trainer.init_weights()
    best_valid_loss = float('inf')

    for epoch in range(config['epochs']):
        
        start_time = time.time()
        
        train_loss, train_bleu = trainer.train(dataObj.train_iterator, CLIP)
        valid_loss, valid_bleu = trainer.evaluate(dataObj.valid_iterator, epoch, dataObj.TRG.vocab, f"emb{config['emb_dim']}_hid{config['hid_dim']}")
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(trainer.model.state_dict(), f"emb{config['emb_dim']}_hid{config['hid_dim']}.model.pt")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train BLEU: {train_bleu:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f}')

        # tune.report(loss = train_loss,
        #             tloss = train_loss, vloss = valid_loss, 
        #             tppl = math.exp(train_loss), vppl = math.exp(valid_loss),
        #             tbleu = train_bleu, vbleu = valid_bleu )

if __name__ == "__main__":
    tune_opt = True
    if tune_opt == True:
        parameters = {
        "emb_dim" : tune.grid_search([256, 512]),
        "hid_dim" : tune.grid_search([512, 1024]),
        "epochs" : 100,
        "device" : torch.device("cuda"),
        "batch_size" : 128
        } 
        ray.init(dashboard_host="0.0.0.0")
        # scheduler = ASHAScheduler(metric="loss", mode="min", max_t=10, grace_period=10)
        # reporter = CLIReporter(metric_columns=["tloss", "vloss", "tppl", "vppn", "tbleu", "vbleu"])
        # reporter.add_metric_column("tloss")
        # reporter.add_metric_column("vloss")
        # reporter.add_metric_column("tppl")
        # reporter.add_metric_column("vppn")
        # reporter.add_metric_column("tbleu")
        # reporter.add_metric_column("vbleu")

        result = tune.run(
            partial(train_model_with_hp),
            resources_per_trial={"cpu": 6, "gpu": 1},
            config=parameters,
            # scheduler=scheduler,
            name="first",
            # progress_reporter=reporter,
            checkpoint_at_end=True,
        )
    else:
        parameters = {
            "emb_dim" : 256,
            "hid_dim" : 512,
            "device" : torch.device('cuda:1'),
            "epochs" : 20,
            "batch_size" : 64
        }
        train_model_with_hp(parameters)
        
