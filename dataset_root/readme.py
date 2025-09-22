"""
标注文件使用： annotate_wavs.py

生成hdf5
python utils/dataset.py pack_waveforms_to_hdf5 --csv_path="dataset_root/metadata/balanced_train_segments.csv" --audios_dir="dataset_root/audios/balanced_train_segments" --waveforms_hdf5_path="dataset_root/hdf5s/waveforms/balanced_train.h5"
生成索引
python utils/create_indexes.py create_indexes --waveforms_hdf5_path="dataset_root/hdf5s/waveforms/balanced_train.h5 --indexes_hdf5_path="dataset_root/hdf5s/indexes/balanced_train.h5"

"""