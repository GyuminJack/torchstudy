import os
import argparse
import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from src.train import *
from src.model import CNN1d
from src.datas import MR


def get_data_obj(data_name):
    data_dict = {"MR": MR()}
    return data_dict[data_name]


def get_model(model_conf):
    cnn_model = CNN1d(**model_conf)
    if model_conf["w2v_init"]:
        if model_conf["embedding_dim"] == 300:
            w2v_path = "/home/jack/torchstudy/1week/0_datas/GoogleNews-vectors-negative300.bin.gz"
            cnn_model._init_embedding_vectors(w2v_path, model_conf["train_vocab"])
        else:
            print("w2v init supports only 300 dimension")
    cnn_model = cnn_model.to(args.device)
    return cnn_model


def main(args):

    ## PARAMETERS
    N_FOLDS = args.n_fold
    OUTPUT_DIM = args.output_dim
    N_EPOCHS = args.epochs
    DATA_NAME = args.data_name
    TRAIN_TEST_RATIO = args.train_test_ratio
    BATCH_SIZE = args.batch_size
    MODEL_TYPE = args.model_type
    EMB_DIM = args.embedding_dim
    FILTERS = args.n_filters
    DROPOUT_RATE = args.dropout_rate
    DEVICE = args.device
    FILTER_SIZES = args.filter_sizes
    W2V_INIT = args.w2v_init
    MAX_NORM_VAL = args.max_norm_val
    SAVE_PATH = args.ck_path

    for model_type in ["multichannel", "static", "non-static"]:
        MODEL_TYPE = model_type
        padlen_list = [-1, 16, 32, 64, 128, 256, 512]
        for pad in padlen_list:
            # GET DATA
            data_obj = get_data_obj(DATA_NAME)
            data_obj.split(split_ratio=TRAIN_TEST_RATIO)
            train_vocabs, train_dataset = data_obj.make_train_datasets(custompad=pad)

            # K-fold Evaluation
            fold = 1
            kfold_acc = []
            kfold_error = []
            kf = KFold(n_splits=N_FOLDS, random_state=12, shuffle=True)
            best_fold_loss = float("inf")
            for train_index, test_index in kf.split(np.arange(len(train_dataset))):
                best_valid_loss = float("inf")
                train_fold_subset = torch.utils.data.Subset(train_dataset, train_index)
                train_fold_iterator = torch.utils.data.DataLoader(
                    train_fold_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    collate_fn=data_obj.collate_pad,
                )

                test_fold_subset = torch.utils.data.Subset(train_dataset, test_index)
                test_fold_iterator = torch.utils.data.DataLoader(
                    test_fold_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    collate_fn=data_obj.collate_pad,
                )

                model_conf = {
                    "model_type": MODEL_TYPE,
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

                model = get_model(model_conf)
                optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
                criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)

                for epoch in range(N_EPOCHS):
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
                print(f"{fold}-fold Loss: {valid_loss:.3f} |  Acc: {valid_acc*100:.2f}%")

                if valid_loss < best_valid_loss and valid_loss < best_fold_loss:
                    best_valid_loss = valid_loss
                    best_fold_loss = valid_loss
                    os.makedirs(SAVE_PATH, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(SAVE_PATH, f"CNN_BEST_{MODEL_TYPE}_{pad}.pt"),
                    )
                kfold_acc.append(valid_acc)
                kfold_error.append(valid_loss)
                fold += 1
            print(f"MODEL_TYPE : {MODEL_TYPE} | CUSTOM PAD_LEN : {pad} |  K-fold Average Loss: {np.mean(kfold_error):.3f} | Average Acc: {np.mean(kfold_acc)*100:.2f}%")

            # If exist Testset  ... continuing inference (not implementation)
            if TRAIN_TEST_RATIO != 1.0:
                test_dataset = data_obj.make_test_datasets()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["static", "non-static", "multichannel"],
        default="non-static",
    )
    parser.add_argument("--w2v_init", type=bool, default=True)
    parser.add_argument("--data_name", type=str, default="MR")
    parser.add_argument("--train_test_ratio", type=float, default=1.0)
    parser.add_argument("--ck_path", type=str, default="./checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--n_filters", type=int, default=100)
    parser.add_argument("--filter_sizes", type=list, default=[3, 4, 5])
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--max_norm_val", type=int, default=3)
    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    import time

    st = time.time()
    main(args)
    print(f"python3 main.py result : {time.time()-st}s")
