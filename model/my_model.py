# import necessary libraries
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, AveragePooling2D, Flatten, Dense, ZeroPadding2D
import tensorflow as tf
from processing import train_data_processing



class model_build:
    def __init__(self):
        self.model = tf.keras.Sequential([
                InputLayer(input_shape=(224, 224, 3)),  # 입력 이미지 크기를 (300, 300, 3)으로 변경
                ZeroPadding2D(padding=(1, 1)),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                AveragePooling2D(pool_size=(2, 2)),
                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                AveragePooling2D(pool_size=(2, 2)),
                Conv2D(32, kernel_size=(3, 3), activation='relu'),
                AveragePooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),  # Dense 레이어의 뉴런 수를 128로 증가
                Dense(19, activation='softmax')  # Softmax activation for multi-class classification
            ])
        self._build()
        self.train()
        self.save_load_model()
        
    def _build(self):
        # 여기서 compile
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

    def train(self):
        # model.fit
        batch_size = 32
        labels_idx, final_images, final_labels_one_hot = train_data_processing()
        print('모델 학습 시작!!')
        history = self.model.fit(final_images, final_labels_one_hot, batch_size=batch_size, epochs=1)
        self.save_load_model()
        return labels_idx

    def save_load_model(self):
        # 모델 저장
        self.model.save('painting_classification_f.h5')