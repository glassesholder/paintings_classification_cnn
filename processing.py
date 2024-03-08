from settings import data_folder, data_forder2
import os
import glob
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler

# 이미지 크기를 조정해주는 함수
def resize_image(image, target_height, target_width):
    # 이미지 크기를 조정합니다.
    resized_image = cv2.resize(image, (target_width, target_height))
    
    
    return resized_image




def train_data_processing():
    data = []

    # 훈련데이터 사진의 이름과 종류, 주소를 데이터프레임으로 만들어 봅시다.

    # 각 폴더의 이름을 하나씩 분석합니다.
    for folder_name in os.listdir(data_folder):
        
        #이미지를 가져올 폴더의 주소를 지정합시다.
        folder_path = os.path.join(data_folder, folder_name)
        
        # 폴더 내의 PNG 파일을 수집합니다.
        png_files = glob.glob(os.path.join(folder_path, '*.png'))
        
        # 각 png 파일을 보면서 이름, 종류, 주소를 확인합니다.
        for png_file in png_files:
            file_name = os.path.basename(png_file)
            name = folder_name + '/' + file_name.split('.')[0]  # 파일 이름 생성
            type = folder_name  # 폴더 이름을 type으로 설정
            data.append({'name': name, 'type': type, 'file_path': png_file.replace("\\","/")})  # 데이터 리스트에 추가

    # 데이터 프레임 생성합니다.
    df = pd.DataFrame(data)

    type_counts = df['type'].value_counts()
    target_types = [type_ for type_ in type_counts.index if type_counts[type_] < type_counts.max()]

    # 오버 샘플링을 위해 샘플링 전략 선택 (이 예시에서는 RandomOverSampler 사용)
    oversample = RandomOverSampler(sampling_strategy={target_type: type_counts.max() for target_type in target_types}, random_state=42)

    # 오버 샘플링을 적용하여 새로운 데이터프레임 생성
    X_resampled, y_resampled = oversample.fit_resample(df.drop(columns=['type']), df['type'])

    # 새로운 데이터프레임 생성 (오버 샘플링된 데이터와 원래 데이터 결합)
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=df.drop(columns=['type']).columns), pd.Series(y_resampled, name='type')], axis=1)

    df_resampled


    # 각각의 사진과 그 사진에 대한 타입을 딕셔너리로 저장합니다.
    data_dict = {}

    for key, val in zip(df_resampled["name"], df_resampled["type"]):
        data_dict[key] = val


    # 각각의 타입에 대한 정수 인코딩을 진행합니다.
    labels = df["type"].unique()
    ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    labels_idx = dict(zip(labels,ids))

    # 최종 훈련데이터 이미지의 데이터 추출할 리스트 및 분류 결과 추출할 리스트
    final_images = []
    final_labels = []
    count = 0


    for file in df_resampled['file_path']:
        file = str(file)
        count += 1
        img_array = np.fromfile(file, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        folder_name, file_name = os.path.split(file)
        # 폴더 이름에서 "train/"을 제거하여 일반화된 폴더 이름 추출
        generalized_folder_name = folder_name.split("train/")[1]

        img = resize_image(img, 224, 224)

        # 결과 출력
        label =  generalized_folder_name + "/" + file_name.split(".")[0]
        label = labels_idx[data_dict[label]]
        # append img in final_images list
        final_images.append(np.array(img))
        # append label in final_labels list
        final_labels.append(np.array(label))

    # 작은 배치로 나누어 처리한 이미지를 합칩니다.
    final_images = np.array(final_images, dtype = np.float32)/255.0
    final_labels = np.array(final_labels, dtype = np.int8).reshape(26695, 1)
    final_labels_one_hot = to_categorical(final_labels, num_classes=19)
    print('훈련 데이터 전처리 끝!')

    return labels_idx, final_images, final_labels_one_hot




def test_data_processing():
    data2 = []

    png_files2 = glob.glob(os.path.join(data_forder2, '*.png'))

    for png_file in png_files2:
            file_name = os.path.basename(png_file)
            name = 'TEST' + '_' + file_name.split('.')[0]  # 파일 이름 생성
            data2.append({'name': name, 'file_path': png_file.replace("\\","/")})  # 데이터 리스트에 추가


    # 데이터 프레임 생성
    df2 = pd.DataFrame(data2)


    test_images = []
    count = 0

    for file in df2['file_path']:
        file = str(file)
        count += 1
        img_array = np.fromfile(file, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = resize_image(img, 224, 224)
        # append img in final_images list
        test_images.append(np.array(img))


    test_images = np.array(test_images, dtype = np.float32)/255.0
    print('테스트 데이터 전처리 끝!')

    return test_images, df2