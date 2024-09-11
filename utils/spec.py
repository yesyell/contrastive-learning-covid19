import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd, HTML, display
import os

# # for Google Colab
# from google.colab import drive
# drive.mount('/content/drive')

import glob
import time

# 파일 진행 상황을 보기 위함
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

# 전체 path에서 어떤 sound인지 추출
def get_soundName(file):
    path=file.split('/')[7].split('.')[0]
    return path

# 전체 path에서 화자 id만 추출
def get_id(file):
    path=file.split('/')[6].split('.')[0]
    return path

# 볼륨 조절 함수 (scaling_factor는 0.5로 50% 볼륨 감소)
def change_volume(y, scaling_factor=1.5):
    return y * scaling_factor

# file name 지정
file_list = []

base_path = os.getcwd()
# print(base_path)
for date_name in glob.glob(os.path.join(base_path, 'data/Extracted_data/*')):
  for dict_name in glob.glob(date_name + '/*'):
    file_list.append(glob.glob(dict_name + '/*.wav'))

# 2차원으로 구성된 file_name list를 1차원으로 변경
file_list = sum(file_list, [])

# data를 분리하기 위한 준비
vowel_o = []
vowel_a = []
vowel_e = []
counting_normal = []
counting_fast = []
cough_shallow = []
cough_heavy = []
breathing_shallow = []
breathing_deep = []

# 파일을 각각에 맞추어서 저장
error_count = 0
for file in file_list:
  match (get_soundName(file)):
    case "vowel-o":
      vowel_o.append(file)
    case "vowel-a":
      vowel_a.append(file)
    case "vowel-e":
      vowel_e.append(file)
    case "counting-normal":
      counting_normal.append(file)
    case "counting-fast":
      counting_fast.append(file)
    case "cough-shallow":
      cough_shallow.append(file)
    case "cough-heavy":
      cough_heavy.append(file)
    case "breathing-shallow":
      breathing_shallow.append(file)
    case "breathing-deep":
      breathing_deep.append(file)
    case _:
      print(f"ERROR {file}")
      error_count += 1
  if error_count == 10:
    break

print(f"ERROR COUNT: {error_count}")

# sound 명에 따라 구분한 파일들을 sound_file이라는 dict안에
sound_file = {'vowel_o': vowel_o, 'vowel_a':vowel_a , 'vowel_e': vowel_e, 'counting_normal':counting_normal, 'counting_fast':counting_fast, 'cough_shallow': cough_shallow, 
              'cough_heavy':cough_heavy, 'breathing_shallow':breathing_shallow, 'breathing_deep':breathing_deep}

sound_key = ['vowel_o', 'vowel_a', 'vowel_e', 'counting_normal', 'counting_fast', 'cough_shallow', 'cough_heavy', 'breathing_shallow', 'breathing_deep']

# 파일을 스펙트럼으로 만들어서 이미지 저장
hop_length = 512
count = 1

# # 진행 상황을 나타내기 위함
# out = display(progress(0, 100), display_id=True)

for key in sound_key:
  # print(key)
  for file in sound_file[key]:
    print(f'file = {file}')
    # 각 sound file 별로 data를 load
    data, sample_rate = librosa.load(file)

    # 볼륨이 조정된 오디오
    data_scaled = change_volume(data)

    # mel-spectrogram 적용
    mfccs = librosa.feature.mfcc(y=data_scaled,sr=sample_rate,n_mfcc=30)

    # 이미지 figure
    plt.figure(figsize=(10, 5))
    # 파일의 id
    id = get_id(file)
    # 파일이 저장될 경로
    path = f'{base_path}/data/spec_data/{key}/{id}.jpg'
    plt.savefig(fname=path, bbox_inches='tight', pad_inches=0)
    # out.update(progress(count*100/24712, 100))
    count += 1
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

print('Image Saving Complete')