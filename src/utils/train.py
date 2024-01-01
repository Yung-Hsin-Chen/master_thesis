from config.config import configure_logging
import logging
from src.utils.metrics import get_wer_cer_per_batch
import torch
import time
from src.utils.helpers import write_predictions_to_file, shutdown_logger
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def get_str(processor, logits):
    pred_ids = logits.argmax(-1)
    # print(type(pred_ids))
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return pred_str

def train(model, **kwargs):
# def train(model, language: str, log_file_path: str, model_path: str, max_epochs: int, train_loader, val_loader, optimizer, device, processor, config_name, text_output=False):
    """
    Trains a given model using the specified training and validation data loaders, and implements early stopping based on validation loss.

    Parameters:
    - model (nn.Module): The model to be trained.
    - **kwargs: Additional keyword arguments including:
        - language (str): The language being used for training, used for logging purposes. Defaults to 'English'.
        - log_file_path (str): File path to save the logging information.
        - model_path (str): File path to save the best model based on validation loss.
        - max_epochs (int): The maximum number of epochs to train.
        - train_loader (DataLoader): DataLoader containing the training data.
        - val_loader (DataLoader): DataLoader containing the validation data.
        - optimizer (Optimizer): The optimizer to use for training.
        - device (torch.device): The device (CPU/GPU) to use for training.
        - processor: The processor used for converting model outputs to text (e.g., for calculating WER, CER).
        - config_name (str): The name of the model configuration, used for logging.
        - text_output (bool): If True, saves the predicted and actual text pairs from each batch. Defaults to False.

    Example Usage:
        model = ... # some initialized model
        train(model, 
            language="English", 
            log_file_path="./training.log", 
            model_path="./best_model.pt", 
            max_epochs=10, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            device=device, 
            processor=processor, 
            config_name="TrOCR",
            text_output=True)

    This function performs training on the model for a given number of epochs or until early stopping is triggered due to no improvement in validation loss. It logs training progress and metrics and saves the best model based on the lowest validation loss.

    The function configures logging, runs training and validation loops, calculates Word Error Rate (WER) and Character Error Rate (CER), implements early stopping, and saves the model with the best validation loss.

    Note:
        - The function assumes that `get_str` and `get_wer_cer_per_batch` are defined and accessible.
        - Logging is configured within the function, and the log file is closed after training.
        - The best model is saved during training, and early stopping is based on validation loss.
    """
    language = kwargs.get("language", "English")
    log_file_path = kwargs["log_file_path"]
    model_path = kwargs["model_path"]
    max_epochs = kwargs.get("max_epochs", 50)
    train_loader = kwargs["train_loader"]
    val_loader = kwargs["val_loader"]
    optimizer = kwargs["optimizer"]
    device = kwargs.get("device", torch.device("cpu"))
    processor = kwargs["processor"]
    config_name = kwargs["config_name"]
    text_output = kwargs.get("text_output", False)

    start_time = time.time()
    if text_output:
        text_output_path = log_file_path.replace("log_", "")
    # Configure Logging
    # configure_logging(log_file_path)  
    # logging.getLogger("noisy_library_name").setLevel(logging.WARNING)
    my_logger = configure_logging(script_name="MyScript", log_file_path=log_file_path)
    my_logger.info("Training process started.")
    # Load processor and model
    my_logger.info("Loading %s on %s data.", config_name, language)
    epoch = 0
    early_stopping = False
    lowest_loss = 100000000
    while (early_stopping==False) and (epoch<max_epochs):
        epoch += 1
        # train_loss, train_wer_sum, train_cer_sum = 0, 0, 0
        train_loss = 0
        val_loss, val_wer_sum, val_cer_sum = 0, 0, 0
        val_samples = 0
        # Clear all contents in the text output file
        with open(text_output_path, "w") as file:
            pass
        # Training
        for i, batch in enumerate(train_loader):
            # print("one batch")
            # train_samples += len(batch.items())
            for k,v in batch.items():
                if k in ["pixel_values", "label_emb", "labels", "attention_mask"]:
                    batch[k] = v.to(device)
            model.train()
            outputs = model(pixel_values=batch["pixel_values"], decoder_input_embeddings=batch["label_emb"], decoder_attention_mask=batch["attention_mask"], labels=batch["labels"])
            # print("train output: ", outputs.logits.size())
            # print("train output type: ", outputs.logits.dtype)
            # print(outputs.logits)
            # print(batch["labels"])
            loss = outputs.loss
            # print("train loss: ", loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            pred_str = get_str(processor, outputs.logits)
            if text_output:
                write_predictions_to_file(pred_str, batch["label_str"], text_output_path)
        # Validation
        with torch.no_grad():
            model.eval() 
            for i, batch in enumerate(val_loader):
                val_samples += len(batch)
                for k,v in batch.items():
                    if k in ["pixel_values", "label_emb", "labels", "attention_mask"]:
                        batch[k] = v.to(device)
                outputs = model(pixel_values=batch["pixel_values"], decoder_input_embeddings=batch["label_emb"], decoder_attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                # print("val loss: ", loss)
                val_loss += loss.detach()
                pred_str = get_str(processor, outputs.logits)
                cer, wer = get_wer_cer_per_batch(pred_str, batch["label_str"])
                if text_output:
                    write_predictions_to_file(pred_str, batch["label_str"], text_output_path)
                val_wer_sum += wer
                val_cer_sum += cer
            val_wer_score = val_wer_sum/val_samples
            val_cer_score = val_cer_sum/val_samples
        if epoch==1:
            my_logger.info(("{:>10} | {:>13} | {:>13} | {:>15} | {:>15}\n" + " "*29 + "{:>10} | {:>13.4f} | {:>13.4f} | {:>15.4f} | {:>15.4f} ").format("Epoch", "Train Loss", "Val Loss", "Val WER", "Val CER", 1, train_loss, val_loss, val_wer_score, val_cer_score))
        else:
            my_logger.info(("{:>10} | {:>13.4f} | {:>13.4f} | {:>15.4f} | {:>15.4f} ").format(epoch, train_loss, val_loss, val_wer_score, val_cer_score))
        # early stopping
        if val_loss <= lowest_loss:
            lowest_loss = val_loss
            # Save the fine-tuned model
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping = True
            my_logger.info("Early stopping at epoch %d.", epoch)
            my_logger.info("Model saved at epoch %d", epoch-1)
    my_logger.info("Training process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    my_logger.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    shutdown_logger(my_logger)
    return