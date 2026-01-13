import os
import librosa
base_path = 'Quadrotor/260109_outdoors_3m_30m/'
babel = '/m/quadrotor'
start_time = 0
end_time = 4
# 读取路径下所有wav文件完整路径
wav_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.wav')]
print(base_path+" 一共有{}个wav文件".format(len(wav_files)))
for wav_file in wav_files:
    print(wav_file.replace('.wav', '') +', '+str(start_time)+', '+str(end_time)+', "'+babel+'"')

# base_path = 'Quadrotor/251231_dataset/'
# babel = '/m/quadrotor'
# start_time = 0; end_time = 4
# # 读取路径下所有wav文件完整路径
# wav_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.wav')]
# print(base_path+" 一共有{}个wav文件".format(len(wav_files)))
# name_list = [os.path.basename(name).replace('.wav', '') for name in wav_files]
# sort_name_index = sorted(range(len(name_list)), key=lambda k: name_list[k])
# (waveform, _) = librosa.core.load(wav_files[sort_name_index[0]], sr=8000, mono=True)
# step = len(waveform) // 8000
# for index in sort_name_index:
#     for i in range(0, step-end_time+start_time, end_time-start_time):
#         print(wav_files[index].replace('.wav', '') +', '+str(i)+', '+str(i+end_time-start_time)+', "'+babel+'"')
