from ml4co_kit import TSPDataGenerator, ATSPDataGenerator

tsp_data_lkh = TSPDataGenerator(
    num_threads=8,
    nodes_num=50,
    data_type="uniform",
    solver="LKH",
    train_samples_num=16, # 1.28M in practice
    val_samples_num=8,
    test_samples_num=0,
    save_path="dir/to/save"
)

atsp_data_lkh = ATSPDataGenerator(
    num_threads=8,
    nodes_num=50,
    data_type="uniform",
    solver="LKH",
    train_samples_num=16, # 1.28M in practice
    val_samples_num=8,
    test_samples_num=0,
    save_path="dir/to/save"
)

# generate supervised training data of TSP & ATSP for MatDIFFNet
tsp_data_lkh.generate()
atsp_data_lkh.generate()
