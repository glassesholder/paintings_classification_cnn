import click
from controller import main
import time

@click.command()
@click.option('-s', '--start')
def start_pipeline(start):
    print("모델 예측 시작!!!")
        
    main()

    time.sleep(2)




if __name__ == '__main__':
    start_pipeline()