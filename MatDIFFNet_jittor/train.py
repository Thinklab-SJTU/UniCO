from src import *
import time

if __name__ == "__main__":
    timestamp = time.time() 
    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))

    # Create MatDIFFNetModel
    model = MatDIFFNetModel(
        env = MatDIFFNetEnv(
            nodes_num=50,
            train_data_size=32000,
            data_type=["TSP", "ATSP", "HCP", "SAT", "VC"],
            tsp_train_path="/mnt/nas-new/home/majiale/Share/Dataset/COExpander/Train/TSP/TSP50_Uniform_Concorde/tsp50_uniform_128k_1.txt",
            tsp_val_path="/mnt/nas-new/home/panwenzheng/MetaGNCO-tmp/MetaGNCO/test_dataset/tsp/tsp50_concorde_5.68759.txt",
            atsp_train_path="/mnt/nas-new/home/majiale/Share/Dataset/COExpander/Train/ATSP/ATSP50_Uniform_LKH-500/atsp50_uniform_32k_1.txt",
            atsp_val_path="/mnt/nas-new/home/panwenzheng/MetaGNCO-tmp/MetaGNCO/test_dataset/atsp/atsp50_uniform_lkh_1000_1.55448.txt",
            val_samples=128,
            train_batch_size=16,
            mode="train",
        ),
        encoder=GNNEncoder(
            num_layers=12,
            hidden_dim=256,
            output_channels=2,
            time_embed_flag=True
        ),
        decoder=MatDIFFNetDecoder(use_2opt=False),
        max_epochs=50,
        val_evry_n_epochs=1,
        ckpt_save_dir=f"./result/{formatted_time[-8:]}/",
        use_cuda=True,
    )
    
    # Train MatDIFFNetModel
    model.model_train()