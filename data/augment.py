# 라이브러리 import
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd


# 노이즈 추가 함수
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    augmented_data = augmented_data.astype(type(y[0]))  # 원 신호와 같은 타입으로 변환
    return augmented_data


# 피치 변경 함수
def pitch_shift(y, sr, n_steps=2):
    try:
        # librosa.effects.pitch_shift 함수 호출 시 y, sr, n_steps 인자 전달
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"Error in pitch shifting: {e}")
        return y
    
    
# 시간 왜곡 함수 (rate는 늘리거나 줄이는 비율, 1.5는 50% 빠르게)
def time_stretch(y, rate=1.5):
    return librosa.effects.time_stretch(y, rate=rate)


# 신호 일부를 무작위로 제거하는 함수
def random_crop(y, crop_fraction=0.1):
    num_samples_to_remove = int(len(y) * crop_fraction)
    start = np.random.randint(0, len(y) - num_samples_to_remove)
    y_cropped = np.concatenate((y[:start], y[start + num_samples_to_remove:]))
    return y_cropped