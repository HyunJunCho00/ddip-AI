import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import re
import json
import joblib
from datetime import datetime

# ===========================
# 모델 클래스 정의
# ===========================
class EnhancedTFLitePricePredictionModel:
    def __init__(self):
        self.max_vocab_size = 2500
        self.max_sequence_length = 80
        self.embedding_dim = 48
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.max_vocab_size,
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.difficulty_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()

        self.keyword_rules = {
            '상': ['애플펜슬','아이패드','노트북','폰','지갑','카드','갤럭시탭','에어팟','워치','카메라','맥북','갤럭시버즈','아이폰','태블릿','갤럭시북','블루투스 이어폰','아이팟','고프로','아이디카드','학생증 카드'],
            '중': ['충전기','우산','텀블러','전공 책','필통','모자','보조배터리','마우스','학생증','목도리','에코백','안경','안경집','USB','책','노트','이어폰','헤어밴드','장갑','볼펜','연필 케이스','명찰'],
            'loss_keywords': ['두고','놓고','잃어','분실','찾아','남겨','흘렸','떨어뜨렸','어딘가에 놔두고','분실한 것 같','기억이 안 나','안 가져왔','어디 뒀는지 모르겠','두고 온 것 같','안 챙겼','깜빡하고 놓고']
        }

        self.base_prices = {'하':500,'중':1500,'상':4000}
        self.combined_model = None

    def preprocess_text(self, title, content):
        text = f"{title} {content}".strip()
        text = re.sub(r'[ㅠㅜㅋㅎ]+','', text)
        text = re.sub(r'[!?]{2,}', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    def keyword_based_difficulty(self, text):
        text_lower = text.lower()
        has_loss = any(k in text_lower for k in self.keyword_rules['loss_keywords'])
        if has_loss:
            if any(k in text_lower for k in self.keyword_rules['상']):
                return '상'
            elif any(k in text_lower for k in self.keyword_rules['중']):
                return '중'
            else:
                return '중'
        return '하'

    def encode_time_cyclical(self, hours):
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        return hour_sin, hour_cos

    def prepare_training_data(self, df):
        texts = [self.preprocess_text(r['요청글 제목'], r['요청글 내용']) for _, r in df.iterrows()]
        self.tokenizer.fit_on_texts(texts)
        text_seq = self.tokenizer.texts_to_sequences(texts)
        text_padded = keras.preprocessing.sequence.pad_sequences(
            text_seq, maxlen=self.max_sequence_length, padding='post', truncating='post'
        ).astype(np.int32)

        difficulties = self.difficulty_encoder.fit_transform(df['난이도'])
        weathers = self.weather_encoder.fit_transform(df['날씨'])

        hours = df['요청 시각'].values
        weekends = (df['주말여부']=='주말').astype(int).values
        hour_sin, hour_cos = self.encode_time_cyclical(hours)

        loss_keyword = np.array([[int(any(k in t for k in self.keyword_rules['loss_keywords'])),
                                  int(any(k in t for k in self.keyword_rules['상']))] for t in texts], dtype=np.float32)

        X = {
            'text_input': text_padded,
            'hour_sin_input': hour_sin.reshape(-1,1).astype(np.float32),
            'hour_cos_input': hour_cos.reshape(-1,1).astype(np.float32),
            'weather_input': weathers.reshape(-1,1).astype(np.int32),
            'weekend_input': weekends.reshape(-1,1).astype(np.float32),
            'keyword_input': loss_keyword
        }

        y = {
            'difficulty_output': difficulties,
            'price_output': df['최종가격'].values
        }
        return X, y

    def build_model(self):
        text_input = layers.Input(shape=(self.max_sequence_length,), name='text_input', dtype='int32')
        hour_sin_input = layers.Input(shape=(1,), name='hour_sin_input', dtype='float32')
        hour_cos_input = layers.Input(shape=(1,), name='hour_cos_input', dtype='float32')
        weather_input = layers.Input(shape=(1,), name='weather_input', dtype='int32')
        weekend_input = layers.Input(shape=(1,), name='weekend_input', dtype='float32')
        keyword_input = layers.Input(shape=(2,), name='keyword_input', dtype='float32')

        # 텍스트 임베딩
        text_embed = layers.Embedding(self.max_vocab_size, self.embedding_dim, input_length=self.max_sequence_length)(text_input)
        text_conv1 = layers.Conv1D(128,3,activation='relu',padding='same')(text_embed)
        text_conv2 = layers.Conv1D(64,3,activation='relu',padding='same')(text_conv1)
        text_pool = layers.GlobalMaxPooling1D()(text_conv2)
        text_dropout = layers.Dropout(0.3)(text_pool)

        weather_embed = layers.Embedding(len(self.weather_encoder.classes_),4)(weather_input)
        weather_flat = layers.Flatten()(weather_embed)

        time_features = layers.concatenate([hour_sin_input, hour_cos_input, weekend_input, keyword_input])
        time_dense = layers.Dense(8,activation='relu')(time_features)

        combined = layers.concatenate([text_dropout, weather_flat, time_dense])
        hidden1 = layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(combined)
        hidden2 = layers.Dense(32,activation='relu')(hidden1)
        hidden2_dropout = layers.Dropout(0.4)(hidden2)

        difficulty_output = layers.Dense(len(self.difficulty_encoder.classes_), activation='softmax', name='difficulty_output')(hidden2)
        price_output = layers.Dense(1, activation='linear', name='price_output')(hidden2)

        self.combined_model = keras.Model(
            inputs=[text_input, hour_sin_input, hour_cos_input, weather_input, weekend_input, keyword_input],
            outputs=[difficulty_output, price_output]
        )

        self.combined_model.compile(
            optimizer='adam',
            loss={'difficulty_output':'sparse_categorical_crossentropy','price_output':'mse'},
            loss_weights={'difficulty_output':0.2,'price_output':0.8},
            metrics={'difficulty_output':'accuracy','price_output':'mae'}
        )
        return self.combined_model

    def train_model(self, df, epochs=30, validation_split=0.2, batch_size=64):
        X, y = self.prepare_training_data(df)
        self.build_model()
        self.combined_model.summary()
        history = self.combined_model.fit(
            X, y, epochs=epochs, validation_split=validation_split, batch_size=batch_size,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                       keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5)],
            verbose=1
        )
        return history

    def predict_price_from_text(self, title, content, weather, hour, is_weekend):
        text = self.preprocess_text(title, content)
        text_seq = self.tokenizer.texts_to_sequences([text])
        text_padded = keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=self.max_sequence_length, padding='post', truncating='post')

        try:
            weather_encoded = self.weather_encoder.transform([weather])[0]
        except:
            weather_encoded = 0

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        weekend_encoded = 1 if is_weekend=='주말' else 0

        loss_keyword = int(any(k in text for k in self.keyword_rules['loss_keywords']))
        high_keyword = int(any(k in text for k in self.keyword_rules['상']))

        inputs = {
            'text_input': text_padded,
            'hour_sin_input': np.array([[hour_sin]],dtype=np.float32),
            'hour_cos_input': np.array([[hour_cos]],dtype=np.float32),
            'weather_input': np.array([[weather_encoded]],dtype=np.int32),
            'weekend_input': np.array([[weekend_encoded]],dtype=np.float32),
            'keyword_input': np.array([[loss_keyword, high_keyword]],dtype=np.float32)
        }

        pred = self.combined_model.predict(inputs, verbose=0)
        diff_probs = pred[0][0]
        price_pred = pred[1][0][0]

        diff_idx = np.argmax(diff_probs)
        diff = self.difficulty_encoder.inverse_transform([diff_idx])[0]
        confidence = diff_probs[diff_idx]

        # 난이도 기반 보정
        if high_keyword and loss_keyword:
            diff = '상'
            confidence = max(confidence, 0.9)
        elif confidence < 0.6:
            diff = self.keyword_based_difficulty(text)

        # 가격 보정: 난이도별 최소 가격 보장
        min_price = self.base_prices.get(diff, 500)
        final_price = max(price_pred, min_price)
        final_price = round(final_price / 100) * 100  # 100원 단위 반올림
        return {'price': int(final_price), 'difficulty': diff, 'confidence': float(confidence), 'raw_prediction': float(price_pred)}


    def evaluate_model(self, df):
        X, y = self.prepare_training_data(df)
        pred = self.combined_model.predict(X)
        diff_pred = np.argmax(pred[0], axis=1)
        price_pred = pred[1].flatten()

        acc = accuracy_score(y['difficulty_output'], diff_pred)
        r2 = r2_score(y['price_output'], price_pred)
        mae = mean_absolute_error(y['price_output'], price_pred)

        print(f"난이도 분류: {acc:.4f}, 가격 R²: {r2:.4f}, MAE: {mae:.2f}")
        return {'difficulty_acc':acc,'price_r2':r2,'price_mae':mae}

    def convert_to_tflite(self, path='price_model.tflite', quantize=False):
        if self.combined_model is None:
            raise ValueError("모델 학습 후 변환 가능")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.combined_model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(path,'wb') as f:
            f.write(tflite_model)
        print(f"TFLite 저장: {path}")
        return path

# ===========================
# 데이터 로드
# ===========================
try:
    df = pd.read_excel('generated_training_data.xlsx')
    print(f"데이터 로드: {len(df)}개 샘플")
except FileNotFoundError:
    print("데이터 없음: generated_training_data.xlsx")
    exit()

# ===========================
# 모델 학습
# ===========================
model = EnhancedTFLitePricePredictionModel()
history = model.train_model(df, epochs=30, validation_split=0.2, batch_size=64)

# ===========================
# 평가
# ===========================
metrics = model.evaluate_model(df)

# ===========================
#  TFLite 변환
# ===========================
tflite_path = model.convert_to_tflite('price_prediction_model.tflite', quantize=True)

print(f"\nTFLite 모델: {tflite_path}")
print(f"최종 평가: {metrics}")

with open('tokenizer_word_index.json','w',encoding='utf-8') as f:
    json.dump(model.tokenizer.word_index, f, ensure_ascii=False)
