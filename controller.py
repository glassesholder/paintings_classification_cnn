from model.load import loader
from processing import train_data_processing, test_data_processing, resize_image
from model.my_model import model_build



def main():
    model_build()
    loader()



