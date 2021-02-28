import os
import argparse
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from src.train import *
from src.model import CNN1d
from src.datas import MR, NSMCDataset
from functools import partial
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from gensim.models import KeyedVectors


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_data_obj(data_name):
    data_dict = {
        "MR": MR(),
        "NSMC_train": NSMCDataset("Train"),
        "NSMC_test": NSMCDataset("Test"),
    }
    return data_dict[data_name]


def get_model_init_vectors(model_conf, ray_w2v_id):
    cnn_model = CNN1d(**model_conf)
    if model_conf["w2v_init"]:
        if model_conf["embedding_dim"] == 300:
            if ray_w2v_id is not None:
                cnn_model._ray_embedding_vectors(ray_w2v_id, model_conf["train_vocab"])
            else:
                w2v_path = "/home/jack/torchstudy/1week/0_datas/GoogleNews-vectors-negative300.bin.gz"
                cnn_model._init_embedding_vectors(w2v_path, model_conf["train_vocab"])
        else:
            print("w2v init supports only 300 dimension")
    else:
        model_conf["train_vocab"].init_vectors(model_conf["embedding_dim"])
        cnn_model._init_embedding_vectors(w2v_path=None, my_vocabs=model_conf["train_vocab"])
    cnn_model = cnn_model.to(args.device)
    return cnn_model


def store_obj(opt):
    if opt == True:
        w2v_path = "/home/jack/torchstudy/1week/0_datas/GoogleNews-vectors-negative300.bin.gz"
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        _id = ray.put(w2v_model)
        return _id
    else:
        return None


def train_model_with_hp(config):

    # GET DATA
    if DATA_NAME == "NSMC":
        data_obj = get_data_obj("NSMC_train")
        train_vocabs, train_dataset = data_obj.make_train_datasets(custompad=config["pad"])
    elif DATA_NAME == "MR":
        data_obj = get_data_obj(DATA_NAME)
        data_obj.split(split_ratio=TRAIN_TEST_RATIO)
        train_vocabs, train_dataset = data_obj.make_train_datasets(custompad=config["pad"])

    # K-fold Evaluation
    fold = 1
    kfold_acc = []
    kfold_error = []
    kf = KFold(n_splits=N_FOLDS, random_state=12, shuffle=True)
    best_fold_loss = float("inf")

    ray_put_opt = RAY_W2V_PUT
    ray_w2v_id = store_obj(ray_put_opt)
    for train_index, test_index in kf.split(np.arange(len(train_dataset))):

        best_valid_loss = float("inf")

        batch_size = config["batch_size"]

        train_fold_subset = torch.utils.data.Subset(train_dataset, train_index)
        train_fold_iterator = torch.utils.data.DataLoader(
            train_fold_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_obj.collate_pad,
        )

        test_fold_subset = torch.utils.data.Subset(train_dataset, test_index)
        test_fold_iterator = torch.utils.data.DataLoader(
            test_fold_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_obj.collate_pad,
        )

        model_conf = {
            "model_type": config["model_type"],
            "vocab_size": len(train_vocabs),
            "embedding_dim": EMB_DIM,
            "n_filters": FILTERS,
            "filter_sizes": FILTER_SIZES,
            "output_dim": OUTPUT_DIM,
            "dropout_rate": DROPOUT_RATE,
            "pad_idx": train_vocabs.vocab_dict["<PAD>"],
            "w2v_init": W2V_INIT,
            "train_vocab": train_vocabs,
        }

        model = get_model_init_vectors(model_conf, ray_w2v_id)
        optimizer = optim.Adadelta(model.parameters(), lr=config["lr"], rho=config["rho"], eps=1e-06)
        criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)

        for epoch in range(config["epoch"]):
            train_loss, train_acc = train(
                model,
                train_fold_iterator,
                optimizer,
                criterion,
                apply_max_norm=True,
                max_norm_val=MAX_NORM_VAL,
                device=DEVICE,
            )
        valid_loss, valid_acc = evaluate(model, test_fold_iterator, criterion, device=DEVICE)

        if valid_loss < best_valid_loss and valid_loss < best_fold_loss:
            best_valid_loss = valid_loss
            best_fold_loss = valid_loss
            os.makedirs(SAVE_PATH, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    "/home/jack/torchstudy/1week/2_refactoring/checkpoint",
                    f"CNN_BEST_{config['model_type']}_{config['pad']}.pt",
                ),
            )
        kfold_acc.append(valid_acc)
        kfold_error.append(valid_loss)
        fold += 1

    tune.report(loss=np.mean(kfold_error), accuracy=np.mean(kfold_acc) * 100)
    print(f"MODEL_TYPE : {config['model_type']} | CUSTOM PAD_LEN : {config['pad']} |  K-fold Average Loss: {np.mean(kfold_error):.3f} | Average Acc: {np.mean(kfold_acc)*100:.2f}%")

    # If exist Testset  ... continuing inference (not implementation)
    if TRAIN_TEST_RATIO != 1.0:
        test_dataset = data_obj.make_test_datasets()


if __name__ == "__main__":
    global N_FOLDS
    global OUTPUT_DIM
    global N_EPOCHS
    global DATA_NAME
    global TRAIN_TEST_RATIO
    global BATCH_SIZE
    global EMB_DIM
    global FILTERS
    global DROPOUT_RATE
    global DEVICE
    global FILTER_SIZES
    global W2V_INIT
    global MAX_NORM_VAL
    global SAVE_PATH
    global RAY_W2V_PUT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["static", "non-static", "multichannel"],
        default="non-static",
    )
    parser.add_argument("--w2v_init", type=boolean_string, default=False)
    parser.add_argument("--data_name", type=str, default="MR")
    parser.add_argument("--train_test_ratio", type=float, default=1.0)
    parser.add_argument("--ck_path", type=str, default="./checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--n_filters", type=int, default=100)
    parser.add_argument("--filter_sizes", type=list, default=[3, 4, 5])
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--max_norm_val", type=int, default=3)
    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ray_w2v_put", type=boolean_string, default=False)
    args = parser.parse_args()

    ## PARAMETERS
    N_FOLDS = args.n_fold
    OUTPUT_DIM = args.output_dim
    N_EPOCHS = args.epochs
    DATA_NAME = args.data_name
    TRAIN_TEST_RATIO = args.train_test_ratio
    # BATCH_SIZE = args.batch_size
    # MODEL_TYPE = args.model_type
    EMB_DIM = args.embedding_dim
    FILTERS = args.n_filters
    DROPOUT_RATE = args.dropout_rate
    DEVICE = args.device
    FILTER_SIZES = args.filter_sizes
    W2V_INIT = args.w2v_init
    MAX_NORM_VAL = args.max_norm_val
    SAVE_PATH = args.ck_path
    RAY_W2V_PUT = args.ray_w2v_put

    hp_MODEL_TYPE = ["non-static"]  # ['static','non-static','multichannel']
    hp_PAD = [-1]  # [-1, 16, 32, 64, 128, 256, 512]
    hp_EPOCH = [10, 20, 30, 40]
    hp_batches = [32, 64, 128, 256, 512]

    baysian_config = {
        "model_type": tune.grid_search(hp_MODEL_TYPE),
        "pad": tune.grid_search(hp_PAD),
        "batch_size": tune.grid_search(hp_batches),
        "epoch": tune.grid_search(hp_EPOCH),
        "lr": tune.grid_search([1.0]),
        "rho": tune.grid_search([0.95]),
    }

    ray.init(dashboard_host="0.0.0.0")
    scheduler = ASHAScheduler(metric="loss", mode="min", max_t=10, grace_period=10)
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "time_total_s"])

    result = tune.run(
        partial(train_model_with_hp),
        resources_per_trial={"cpu": 6, "gpu": 1},
        config=baysian_config,
        scheduler=scheduler,
        name="rand_baysian_2",
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )
