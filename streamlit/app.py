import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
from audio import change_volume

# OpenMP 오류 해결을 위한 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# streamlit 웹 배포를 위한 절대경로 포함
def get_absolute_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)

logo_path = get_absolute_path('forapp/logo.jpg')
logo = Image.open(logo_path)

class MoCoV2(nn.Module):
    def __init__(self, base_encoder, dim=128, K=4096, m=0.999, T=0.07):
        super(MoCoV2, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder()
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(self.encoder_q.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        self.encoder_k = base_encoder()
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(self.encoder_k.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_covid", torch.zeros(K, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, covid):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[:batch_size]
            covid = covid[:batch_size]

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_covid[ptr:ptr + batch_size] = covid

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, covid):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        queue = self.queue.clone().detach()
        queue_covid = self.queue_covid.clone().detach()

        neg_idx = (queue_covid != covid.unsqueeze(1)).float()
        l_neg = torch.einsum('nc,ck->nk', [q, queue])
        l_neg = l_neg * neg_idx

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k, covid)

        return logits, labels

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim + 1, 256)  # +1 for COVID_symptoms
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, covid):
        x = torch.cat([x, covid.unsqueeze(1)], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# 모델 로드 함수
@st.cache_resource
def load_model():
    moco = MoCoV2(base_encoder=models.resnet50, K=4096)
    classifier = LinearClassifier(128, num_classes=2)  # Assuming binary classification
    
    moco_path = get_absolute_path('forapp/ckpoint/moco_covid_metadata_best_loss.pth')
    classifier_path = get_absolute_path('forapp/ckpoint/classifier_best_covid.pth')
    
    moco.load_state_dict(torch.load(moco_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict'])
    classifier.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict'])
    
    moco.eval()
    classifier.eval()
    
    return moco, classifier


# 예측 함수 수정
def process_audio_and_predict(audio_array, sample_rate, moco, classifier, respiratory_ailment):
    print("예측 함수 시작")
    try:
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=30)
        print(f"MFCC 생성 완료. Shape: {mfccs.shape}")
        mfccs = np.expand_dims(mfccs, axis=0)
        mfccs = np.repeat(mfccs, 3, axis=0)
        
        mfccs_tensor = torch.from_numpy(mfccs).float().unsqueeze(0)
        print(f"MFCC 텐서 생성 완료. Shape: {mfccs_tensor.shape}")
        
        with torch.no_grad():
            features = moco.encoder_q(mfccs_tensor)
        print(f"특징 추출 완료. Shape: {features.shape}")
        
        with torch.no_grad():
            outputs = classifier(features, respiratory_ailment)
            _, predicted = torch.max(outputs, 1)
        
        print(f"예측 완료. 결과: {predicted.item()}")
        return predicted.item()
    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
        return None


# Streamlit 앱 시작
st.set_page_config(page_title="COVID-19 호흡음 분석", page_icon="🩺")

st.image(logo, use_column_width=True)
st.title("🕵️ 당신의 호흡음은 코로나를 알고있다!")
st.write('안녕하세요, 저희는 24su deep daiv Medical AI 음파음파 팀 입니다. \n 저희는 **환자의 메타데이터**와 **호흡음 데이터**를 **대조학습**으로 코로나를 진단하는 프로젝트를 진행했습니다.')
st.write('당신의 호흡음을 녹음하고, MFCC로 변환하여 코로나 양성 여부를 예측해 드립니다. 🦠')

st.write()
st.info('📌 **<Demo Page 사용설명서>** 📌'
        '\n 1. 🎙️ 사용자의 호흡음 녹음'
        '\n     - **"START" 버튼**을 클릭하여 녹음을 시작합니다.'
        '\n     - 호흡음을 녹음한 후 "STOP" 버튼을 클릭하여 녹음을 종료합니다.'
        '\n     - **"녹음 종료" 버튼**을 클릭하여 녹음된 오디오를 저장합니다.'
        '\n 2. 👩‍💼 녹음된 호흡음 확인 및 분석'
        '\n     - 저장된 오디오를 재생하여, 제대로 녹음이 되었는지 확인합니다.'
        '\n     - **"녹음 분석" 버튼**을 클릭하여 MFCC로 호흡음을 시각화합니다.'
        '\n 3. 🩺 코로나 증상 선택 및 양/음성 예측'
        '\n     - 코로나 관련 증상이 있다면 선택합니다.'
        '\n     - **"🧑‍⚕️ 호흡음 코로나 상태 예측" 버튼**을 클릭하여 모델의 예측 결과를 확인합니다.'
        '\n 4. 🔄 초기화'
        '\n     - 예측 결과 확인 후 **"처음부터 다시 시작" 버튼**을 클릭하여 모든 데이터를 초기화하고 처음부터 다시 시작할 수 있습니다.'
        )
st.error('🚨 **<주의사항>** 🚨'
         '\n - 데모 결과는 참고용으로만 활용해주세요.'
         '\n - 각 단계를 순서대로 진행해 주세요. 단계를 건너뛰면 오류가 발생할 수 있습니다.'
)
st.markdown('---')

st.subheader("😤 호흡음을 녹음해 주세요!")

# Load the model
moco, classifier = load_model()

# 세션 상태 초기화
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'mfccs' not in st.session_state:
    st.session_state.mfccs = None
if 'mfcc_image' not in st.session_state:
    st.session_state.mfcc_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []

def reset_session_state():
    st.session_state.audio_data = None
    st.session_state.mfccs = None
    st.session_state.mfcc_image = None
    st.session_state.prediction_result = None
    st.session_state.selected_symptoms = []

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if webrtc_ctx.audio_receiver:
    if webrtc_ctx.state.playing:
        st.write("🎙️녹음 중... 마이크에 대고 말씀해 주세요!")
        
    if st.button("녹음 종료"):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        audio_data = []
        for frame in audio_frames:
            audio_data.extend(frame.to_ndarray().flatten())
        st.session_state.audio_data = np.array(audio_data, dtype=np.float32)
        st.write("녹음이 완료되었습니다.")
        st.session_state.mfccs = None
        st.session_state.mfcc_image = None
        st.session_state.prediction_result = None

if st.session_state.audio_data is not None:
    st.audio(st.session_state.audio_data, sample_rate=48000)
    
    if st.button("녹음 분석"):
        st.write("👨‍💼녹음된 오디오를 처리 중...")
        st.markdown("---")

        sample_rate = 48000
        audio_array = st.session_state.audio_data
        data_scaled = change_volume(audio_array)

        st.subheader("💻 MFCC로 표현된 당신의 호흡음은?")
        
        try:
            with st.spinner('MFCC로 호흡음을 변환 중에 있어요...'):
                hop_length = 512
                mfccs = librosa.feature.mfcc(y=data_scaled, sr=sample_rate, n_mfcc=30)
                st.session_state.mfccs = mfccs

                plt.figure(figsize=(10, 5))
                librosa.display.specshow(mfccs, sr=sample_rate, hop_length=hop_length)
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                st.session_state.mfcc_image = buf.getvalue()
                st.image(st.session_state.mfcc_image, caption='MFCC Spectrogram', use_column_width=True)
                plt.close()
            st.success("MFCC 변환 완료")
        except Exception as e:
            st.error(f"MFCC 변환 중 오류 발생: {str(e)}")

        wav_file = BytesIO()
        wavfile.write(wav_file, sample_rate, (audio_array * 32767).astype(np.int16))
        wav_file.seek(0)
        st.download_button(label="📂 호흡음 다운로드", data=wav_file, file_name="respiratory_sounds.wav", mime="audio/wav")
        
    st.write()
    st.subheader("😷 코로나 증상 선택")
    st.write("아래 증상 중 해당하는 증상을 선택해주세요. 해당 하는 증상이 없다면, 선택하지 않아도 됩니다.")
    # 증상 선택 섹션 추가
    symptoms = ['기침', '감기', '설사', '호흡곤란', '열', '과다피로', '근육통', '미각/후각 상실']
    st.session_state.selected_symptoms = st.multiselect('다음과 같은 코로나 증상이 있었나요? (복수 선택 가능)', symptoms, st.session_state.selected_symptoms)

    if st.button("🧑‍⚕️ 호흡음 코로나 상태 예측"):
        if st.session_state.mfccs is not None:
            respiratory_ailment = torch.tensor([1.0]) if st.session_state.selected_symptoms else torch.tensor([0.0])
            prediction = process_audio_and_predict(st.session_state.audio_data, 48000, moco, classifier, respiratory_ailment)
            if prediction is not None:
                st.session_state.prediction_result = prediction
            else:
                st.error("예측 실패")
        else:
            st.error("먼저 '녹음 분석' 버튼을 클릭하여 MFCC를 생성해주세요.")

    if st.session_state.prediction_result is not None:
        st.subheader("📒 호흡음 코로나 상태 예측 결과")
        if st.session_state.mfcc_image:
            st.image(st.session_state.mfcc_image, caption='MFCC Spectrogram', use_column_width=True)
        if st.session_state.prediction_result == 1:
            st.warning("코로나 양성으로 예측됩니다. 의료진과 상담을 권장드립니다.")
        else:
            st.success("코로나 음성으로 예측됩니다. 하지만 의심 증상이 있다면 검사를 받아보세요.")
        
        st.subheader("선택한 증상")
        if st.session_state.selected_symptoms:
            st.write(", ".join(st.session_state.selected_symptoms))
        else:
            st.write("선택한 증상이 없습니다.")

        if st.button("🔄 처음부터 다시 시작"):
            reset_session_state()
            st.rerun()

else:
    st.write("녹음을 시작하려면 'START' 버튼을 클릭하세요.")

st.markdown('---')
st.warning('🤖 **디버그 정보** (실시간 확인용)'
        f'\n - 오디오 데이터 존재: {st.session_state.audio_data is not None}'
        f'\n - MFCC 데이터 존재: {st.session_state.mfccs is not None}'        
        f'\n - 예측 결과: {st.session_state.prediction_result}'
        f'\n - 선택한 증상: {", ".join(st.session_state.selected_symptoms) if st.session_state.selected_symptoms else "없음"}'
        )