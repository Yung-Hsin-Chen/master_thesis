from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor, model
