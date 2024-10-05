import numpy as np
from streamlit_webrtc import AudioProcessorBase


# 볼륨 조절 함수 구현
def change_volume(data, factor=1.5):
    return np.clip(data * factor, -1, 1)

# AudiooProcessor 클래스 구현 (for streamlit-webrtc)
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame):
        self.audio_data.extend(frame.to_ndarray().flatten().tolist())
        return frame

    def get_audio_data(self):
        return np.array(self.audio_data, dtype=np.float32)