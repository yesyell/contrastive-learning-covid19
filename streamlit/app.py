import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from audio import change_volume

# 실험 완료한 모델 import
# from your_model_module import load_model, predict

st.title("🕵️ 당신의 호흡음은 코로나를 알고있다!")
st.write('안녕하세요, 저희는 24su Medical AI 음파음파 팀 입니다. 저희는 메타데이터와 호흡음 데이터를 대조학습으로 코로나를 분류하는 프로젝트를 진행했습니다.')
st.info('**<Demo Page 사용설명서>**'
        '\n 1. 사용자의 호흡음 녹음'
        '\n     - START 이후 표시되는 STOP 버튼을 누르면 프로세스 전체가 종료되니 주의해 주세요!'
        '\n 2. 호흡음을 모델이 이해할 수 있도록 MFCC로 시각화'
        '\n 3. 코로나 예측 🩺'
        )
st.markdown('---')

st.subheader("😤 호흡음을 녹음해 주세요!")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv_queued(self, frames):
        for frame in frames:
            self.audio_data.extend(frame.to_ndarray().flatten().tolist())
        return frames

    def get_audio_data(self):
        return np.array(self.audio_data, dtype=np.float32)

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# 녹음 중 상태를 보여주는 UI
if webrtc_ctx.state.playing:
    st.write("🎙️녹음 중... 마이크에 대고 말씀해 주세요!")
    if webrtc_ctx.audio_processor:
        st.write(f"녹음된 샘플 수: {len(webrtc_ctx.audio_processor.audio_data)}")

# 녹음 종료 버튼
if st.button("녹음 종료 및 분석"):
    if webrtc_ctx.audio_processor and len(webrtc_ctx.audio_processor.audio_data) > 0:
        st.write("👨‍💼녹음된 오디오를 처리 중...")
        st.markdown("---")

        audio_array = webrtc_ctx.audio_processor.get_audio_data()
        sample_rate = 48000  # WebRTC의 일반적인 샘플링 레이트

        # 오디오 데이터 볼륨 조절
        data_scaled = change_volume(audio_array)

        st.subheader("💻 MFCC로 표현된 당신의 호흡음은?")
        
        with st.container():
            with st.spinner('MFCC로 호흡음을 변환 중에 있어요...'):
                # MFCC 계산
                hop_length = 512
                mfccs = librosa.feature.mfcc(y=data_scaled, sr=sample_rate, n_mfcc=30)

                # MFCC 시각화
                plt.figure(figsize=(10, 5))
                librosa.display.specshow(mfccs, sr=sample_rate, hop_length=hop_length)
                
                # 이미지를 BytesIO 객체로 저장
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                st.image(buf, caption='MFCC Spectrogram', use_column_width=True)

                plt.close()  # 메모리 누수 방지를 위해 figure 닫기

        # 오디오 파일 다운로드 링크 제공 및 모델 예측
        col1, col2 = st.columns(2)
        
        with col1:
            # WAV 파일 생성 및 다운로드 링크 제공
            wav_file = BytesIO()
            wavfile.write(wav_file, sample_rate, audio_array)
            wav_file.seek(0)
            st.download_button(label="📂 호흡음 다운로드", data=wav_file, file_name="respiratory_sounds.wav", mime="audio/wav")

        with col2:
            if st.button("🧑‍⚕️ 호흡음 코로나 상태 예측"):
                # 모델 예측 (의사 코드)
                # model = load_model("path_to_your_model")
                # prediction = predict(model, mfccs)
                # with st.spinner('당신의 호흡음을 분석하고 있어요...'):
                #     time.sleep(2)
                # st.success(f"예측 결과: {prediction}")

                st.success("모델 예측 결과가 아마 이쯤에 뜰거에요!")
        
        # 오디오 데이터 초기화
        webrtc_ctx.audio_processor.audio_data.clear()
    else:
        st.write("녹음된 오디오가 없습니다.")
else:
    st.write("녹음을 종료하고 분석하려면 '녹음 종료 및 분석' 버튼을 클릭하세요.")