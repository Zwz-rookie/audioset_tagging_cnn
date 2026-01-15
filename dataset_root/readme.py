"""
标注文件使用： annotate_wavs.py

生成hdf5
python utils/dataset.py pack_waveforms_to_hdf5 --csv_path="dataset_root/metadata/gk_train_segments.csv" --audios_dir="dataset_root/audios/balanced_train_segments" --waveforms_hdf5_path="dataset_hdf5/hdf5s/waveforms/balanced_train.h5"
生成索引
python utils/create_indexes.py create_indexes --waveforms_hdf5_path="dataset_hdf5/hdf5s/waveforms/balanced_train.h5" --indexes_hdf5_path="dataset_hdf5/hdf5s/indexes/balanced_train.h5"
进行训练
python main_epoch train
--data_type=balanced_train
--checkpoint_path=MobileNetV2_Mod.pth
--workspace=dataset_hdf5
--sample_rate=8000
--window_size=1024
--hop_size=320
--mel_bins=64
--fmin=50
--fmax=14000
--model_type=MobileNetV2_Mod
--loss_type=clip_bce
--balanced=balanced
--augmentation=none
--batch_size=32
--learning_rate=1e-3
--resume_iteration=0
--early_stop=1000000
--cuda
测试模型
python inference audio_tagging
--sample_rate=8000
--model_type="MobileNetV2_Mod"
--checkpoint_path="MobileNetV2_Mod.pth"
--audio_path="resources/Channel_2_Pos_1009.wav"
--cuda
"""