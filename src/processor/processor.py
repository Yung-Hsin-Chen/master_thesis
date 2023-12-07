from src.processor.process_image import process_image
from src.processor.process_gt import process_gt

def process_data():
    process_image()
    process_gt()
    return

if __name__=="__main__":
    process_data()
