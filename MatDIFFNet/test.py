from src import *


# Test Settings
TYPE = "atsp50"
MIX_PRETRAIN = False
TWO_OPT = True

# Static
DATASET_DICT = {
    "tsp50": "test_dataset/tsp50.txt",
    "atsp50": "test_dataset/atsp50.txt",
    "hcp50": "test_dataset/hcp50.txt",
    "sat50": "test_dataset/sat50.txt",
    "vc50": "test_dataset/vc50.txt",
    "tsp100": "test_dataset/tsp100.txt",
    "atsp100": "test_dataset/atsp100.txt",
    "hcp100": "test_dataset/hcp100.txt",
    "sat100": "test_dataset/sat100.txt",
}

PRETRAIN_DICT = {
    "tsp50": {
        False: "weights/matdiffnet_tsp50.pt",
        True: "weights/matdiffnet_mix50.pt"
    },
    "atsp50": {
        False: "weights/matdiffnet_atsp50.pt",
        True: "weights/matdiffnet_mix50.pt"
    },
    "hcp50": {
        False: "weights/matdiffnet_hcp50.pt",
        True: "weights/matdiffnet_mix50.pt"
    },
    "sat50": {
        False: "weights/weights50.pt",
        True: "weights/matdiffnet_mix50.pt"
    },
    "vc50": {
        False: "weights/matdiffnet_vc50.pt",
        True: "weights/matdiffnet_mix50.pt"
    },
    "tsp100": {
        False: "weights/matdiffnet_tsp100.pt",
        True: "weights/matdiffnet_mix100.pt"
    },
    "atsp100": {
        False: "weights/matdiffnet_atsp100.pt",
        True: "weights/matdiffnet_mix100.pt"
    },
    "hcp100": {
        False: "weights/matdiffnet_hcp100.pt",
        True: "weights/matdiffnet_mix100.pt"
    },
    "sat100": {
        False: "weights/weights100.pt",
        True: "weights/matdiffnet_mix100.pt"
    }
}


# Main
if __name__ == "__main__":
    solver = MatDIFFNetSolver(
        model=MatDIFFNetModel(
            env=MatDIFFNetEnv(device="cuda"),
            encoder=GNNEncoder(),
            decoder=MatDIFFNetDecoder(use_2opt=TWO_OPT)
        ),
        pretrained_path=PRETRAIN_DICT[TYPE][MIX_PRETRAIN]
    )
    dists = solver.from_txt(DATASET_DICT[TYPE], return_list=True, show_time=True)[0]
    solver.solve(dists=dists, show_time=True)
    solver.evaluate()