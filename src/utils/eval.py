from config.config import configure_logging
import logging
from src.utils.metrics import get_wer_cer_per_batch
import time

def eval(model_name: str, language: str, log_file_path: str, epochs: int, dataloader, load_model, ocr, criterion) -> None:
    start_time = time.time()
    # Configure Logging
    configure_logging(log_file_path)  
    logging.info("Evaluation process started.")
    # Load processor and model
    logging.info("Loading %s on %s data.", model_name, language)
    model = load_model()
    for epoch in epochs:
        wer_sum, cer_sum = 0, 0
        # Iterate over batches in the dataloader
        for batch in dataloader:

            images, labels = batch

            outputs = ocr(model, images)
            loss = criterion(outputs.float(), labels.float())
            wer, cer = get_wer_cer_per_batch(outputs, labels)
            wer_sum += wer
            cer_sum += cer
        wer_score = wer_sum/len(dataloader.dataset)
        cer_score = cer_sum/len(dataloader.dataset)
        logging.info("{:>10} | {:>13.4f} | {:>15.4f} | {:>15.4f} ".format(epoch, loss, wer_score, cer_score))
    logging.info("Evaluation process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    logging.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logging.shutdown()
    return