from config.config import configure_logging
import logging
from src.utils.metrics import get_wer_cer_per_batch
import torch
import time
from src.utils.helpers import write_predictions_to_file, shutdown_logger
import torch.nn as nn
# from captum.attr import IntegratedGradients
# from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import os
from config.config_paths import RESULTS

import config.model_config as cfg
charbert_args = cfg.charbert_dataset_args

criterion = nn.CrossEntropyLoss()

def get_str(processor, logits):
    pred_ids = logits.argmax(-1)
    # print(type(pred_ids))
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return pred_str

def eval(model, **kwargs):
    language = kwargs.get("language", "English")
    log_file_path = kwargs["log_file_path"]
    test_loader = kwargs["test_loader"]
    device = kwargs.get("device", torch.device("cpu"))
    processor = kwargs["processor"]
    config_name = kwargs["config_name"]
    text_output = kwargs.get("text_output", False)
    model_name = kwargs["model_name"]

    start_time = time.time()
    if text_output:
        text_output_path = log_file_path.replace("log_", "")
    # Configure Logging
    # configure_logging(log_file_path)  
    # logging.getLogger("noisy_library_name").setLevel(logging.WARNING)
    my_logger = configure_logging(script_name="MyScript", log_file_path=log_file_path)
    my_logger.info("Evaluation process started.")
    # Load processor and model
    my_logger.info("Loading %s on %s data.", config_name, language)
    model = model.to(device)
    # Initialize Integrated Gradients
    with open(text_output_path, "w") as file:
        pass
    test_samples, test_wer_sum, test_cer_sum = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_loader):
            # Reset the internal state before making a prediction
            # model.decoder.init_decoder_cache(batch_size=1)
            test_samples += batch["pixel_values"].size(0)
            for k,v in batch.items():
                if k in ["pixel_values", "start_ids", "end_ids", "input_ids", "char_input_ids"]:
                    # print(v.size())
                    batch[k] = v.to(device)
            # print(batch["char_input_ids"][:, :5])
            # print(batch["input_ids"])
            if model_name=="trocr":
                generated_ids = model.generate(pixel_values=batch["pixel_values"])
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # print(generated_text)
            if model_name=="seq_models":
                print("label: ", batch["label_str"])
                # print("input_ids", batch["input_ids"])
                # print("start_ids", batch["start_ids"])
                outputs = model(char_input_ids=batch["char_input_ids"], start_ids=batch["start_ids"], end_ids=batch["end_ids"], input_ids=batch["input_ids"])
                # print(outputs)
                predictions = torch.argmax(outputs, dim=2)[0]
                # print(predictions)
                # print("label: \n", batch["label_str"])
                generated_text = charbert_args.tokenizer.decode(predictions)
                print("generated text: \n", generated_text)
            # print(generated_text)
            wer, cer = get_wer_cer_per_batch([generated_text], batch["label_str"])
            # print(cer)
            if text_output:
                write_predictions_to_file([generated_text], batch["label_str"], text_output_path)
            test_wer_sum += wer
            test_cer_sum += cer
        # print("test: ", test_samples)
        test_wer_score = test_wer_sum/test_samples
        test_cer_score = test_cer_sum/test_samples
        my_logger.info(f"Test WER = {test_wer_score}")
        my_logger.info(f"Test CER = {test_cer_score}")
        torch.cuda.empty_cache()

    # Save the figure
    # plt.savefig(os.path.join(RESULTS, "analysis.png"), bbox_inches="tight")
    # plt.close(fig)  # Close the figure to free memory
    my_logger.info("Evaluation process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    my_logger.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    shutdown_logger(my_logger)
    return test_wer_score, test_cer_score