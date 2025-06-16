from src import *
import jittor as jt

# Test Settings
TYPE = "atsp50"
MIX_PRETRAIN = True
TWO_OPT = False

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

PRETRAIN_PATH = "path/to/your/jittor/weight.pkl"

# Main
if __name__ == "__main__":
    jt.flags.use_cuda = True
    solver = MatDIFFNetSolver(
        model=MatDIFFNetModel(
            env=MatDIFFNetEnv(),
            encoder=GNNEncoder(),
            decoder=MatDIFFNetDecoder(use_2opt=TWO_OPT),
            inference_diffusion_steps=20
        ),
        pretrained_path=PRETRAIN_PATH
    )
    dists = solver.from_txt(DATASET_DICT[TYPE], return_list=True, show_time=True)[0]
    solver.solve(dists=dists, show_time=True)
    solver.evaluate()