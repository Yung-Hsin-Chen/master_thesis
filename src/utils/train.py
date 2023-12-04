from config.config import configure_logging
import logging
from src.utils.metrics import get_wer_cer_per_batch
import torch
import time

def train(model_name: str, language: str, log_file_path: str, model_path: str, epochs: int, trainloader, valloader, load_model, ocr, criterion, optimiser):
    start_time = time.time()
    # Configure Logging
    configure_logging(log_file_path)  
    logging.info("Training process started.")
    # Load processor and model
    logging.info("Loading %s on %s data.", model_name, language)
    model = load_model()
    epoch = 0
    early_stopping = False
    lowest_loss = 100000000
    while (early_stopping==False) and (epoch<epochs):
        epoch += 1
        train_loss, train_wer_sum, train_cer_sum = 0, 0, 0
        val_loss, val_wer_sum, val_cer_sum = 0, 0, 0
        # Training
        for batch in trainloader:
            images, labels = batch
            outputs = ocr(model, images)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimiser.step()
            train_loss += loss.detach()
            wer, cer = get_wer_cer_per_batch(outputs, labels)
            train_wer_sum += wer
            train_cer_sum += cer
        train_wer_score = train_wer_sum/len(trainloader.dataset)
        train_cer_score = train_cer_sum/len(trainloader.dataset)
        # Validation
        with torch.no_grad():
            for batch in valloader:
                images, labels = batch
                outputs = ocr(model, images)
                loss = criterion(outputs.float(), labels.float())
                val_loss += loss.detach()
                wer, cer = get_wer_cer_per_batch(outputs, labels)
                val_wer_sum += wer
                val_cer_sum += cer
            val_wer_score = val_wer_sum/len(valloader.dataset)
            val_cer_score = val_cer_sum/len(valloader.dataset)
        logging.info("{:>10} | {:>13.4f} | {:>13.4f} | {:>15.4f} | {:>15.4f} | {:>15.4f} | {:>15.4f} "
                    .format(epoch, train_loss, val_loss, train_wer_score, val_wer_score, train_cer_score, val_cer_score))
        # early stopping
        if val_loss <= lowest_loss:
            lowest_loss = val_loss
            # Save the fine-tuned model
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping = True
            logging.info("Early stopping at epoch %d.", epoch)
            logging.info("Model saved at epoch %d", epoch-1)
    logging.info("Training process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    logging.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logging.shutdown()
    return