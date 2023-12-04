from src.models.trocr import load_model
import os
import time
import logging
from PIL import Image
from src.utils.helpers import load_data
from config.config import configure_logging

def eval_trocr(log_file_path, text_output=False):
    if text_output:
        text_list = []
    configure_logging(log_file_path)
    start_time = time.time()
    logging.info("Evaluation process started.")
    processor, model = load_model()
    data = load_data()
    for image, label in zip(data["GW_image"], data["GW_gt"]):
        image = Image.open(image).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        # Generate text from the model
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        text_list.append((generated_text, label))
    # Open the txt file in append mode
    with open(log_file_path.replace("logs.txt", "output.txt"), "a") as file:
        for text in range(5):
            data_to_write = f"{text[0]}, {text[1]}\n"
            file.write(data_to_write)
    logging.info("Evaluation process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    logging.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logging.shutdown()
    return

if __name__ == "__main__":
    log_file_path = os.path.join(".", "results", "trocr_exp01", "logs.txt")
    eval_trocr(log_file_path, text_output=True)