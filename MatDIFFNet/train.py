from src import *
from ml4co_kit import Trainer


if __name__ == "__main__":
    # Create MatDIFFNetModel
    model = MatDIFFNetModel(
        env = MatDIFFNetEnv(
            nodes_num=50,
            train_data_size=128000,
            data_type=["TSP", "ATSP", "HCP", "SAT", "VC"],
            tsp_train_path="your/train/tsp/txt/path",
            tsp_val_path="your/validation/tsp/txt/path",
            atsp_train_path="your/train/atsp/txt/path",
            atsp_val_path="your/validation/atsp/txt/path",
            val_samples = 16,
            train_batch_size = 4,
            mode="train",
            device="cuda",
        ),
        encoder=GNNEncoder(
            num_layers=12,
            hidden_dim=256,
            output_channels=2,
            time_embed_flag=True
        ),
        decoder=MatDIFFNetDecoder()
    )
    
    # Train MatDIFFNetModel using CUDA
    trainer = Trainer(model=model, devices=[0], max_epochs=20)
    trainer.model_train()