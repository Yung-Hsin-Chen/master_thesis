from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from src.utils.helpers import load_data
from PIL import Image

def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor, model
