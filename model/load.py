from processing import test_data_processing, train_data_processing
import numpy as np
import pandas as pd
from keras.models import load_model





def loader():

    test_images, df2 = test_data_processing()
    loaded_model = load_model('painting_classification_f.h5')
    predictions = loaded_model.predict(test_images)
    
    answer =[]

    labels_idx, final_images, final_labels_one_hot =  train_data_processing()

    for prediction in predictions:
        max_index = np.argmax(prediction)
        type_of_image = list(labels_idx.keys())[max_index]
        answer.append(type_of_image)

    df2['label'] = answer
    df2.drop(['file_path'], axis=1, inplace=True)
    df2.rename(columns={'name': 'id'}, inplace=True)
    df2.to_csv('submission_f.csv', index=False)
    print('업로드 성공!!')