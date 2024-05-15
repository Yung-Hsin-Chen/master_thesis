from config.config import configure_logging
import logging
from src.utils.metrics import get_wer_cer_per_batch
import torch
import time
from src.utils.helpers import write_predictions_to_file, shutdown_logger
import torch.nn as nn
import torch.nn.functional as F
# from captum.attr import IntegratedGradients
# from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import os
from config.config_paths import RESULTS
from torch.autograd import profiler
import config.model_config as cfg
from src.models.charbert_model import initialise_charbert_model
import shutil

criterion = nn.CrossEntropyLoss()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def get_str(processor, logits):
    pred_ids = logits.argmax(-1)
    # print(type(pred_ids))
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return pred_str

def filter_string(input_string):
    # Define the allowed characters: 26 lowercase letters, 10 digits, and space
    allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789 ')
    
    # Filter the string to include only characters in the allowed set
    filtered_string = ''.join(char for char in input_string.lower() if char in allowed_chars)
    
    return filtered_string

def train(model, freeze_mode, layers, **kwargs):
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
    test_loader = kwargs["test_loader"]
    optimizer = kwargs["optimizer"]
    device = kwargs.get("device", torch.device("cpu"))
    processor = kwargs["processor"]
    config_name = kwargs["config_name"]
    text_output = kwargs.get("text_output", False)
    lr = kwargs["lr"]
    decay_rate = kwargs["weight_decay"]
    model_name = kwargs["model_name"]
    text_file = kwargs["text_file"]
    # print("MODEL NAME: ", model_name)
    start_time = time.time()
    if text_output:
        text_output_path = log_file_path.replace("log_", "")
    # Configure Logging
    # configure_logging(log_file_path)  
    # logging.getLogger("noisy_library_name").setLevel(logging.WARNING)
    my_logger = configure_logging(script_name="MyScript", log_file_path=log_file_path)
    my_logger.info("Training process started.")
    # Load processor and model
    my_logger.info("Loading %s on %s data.", str(model_name).replace("_", " "), language)
    device_name = "CPU" if device==torch.device("cpu") else "GPU"
    my_logger.info("Training on %s.", device_name)
    # formatted_lr = f"{lr:.15f}".rstrip("0").rstrip(".")
    formatted_lr = "{:.0e}".format(lr)
    # formatted_decay_rate = f"{decay_rate:.15f}".rstrip("0").rstrip(".")
    formatted_decay_rate = "{:.0e}".format(decay_rate)
    my_logger.info("Learning rate = %s; Weight decay = %s", formatted_lr, formatted_decay_rate)
    epoch = 0
    early_stopping = False
    lowest_loss = 100000000
    save_epoch = 1
    lowest_cer = 1000000000
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     # Wrap the model with DataParallel
    #     model = nn.DataParallel(model)
    # model.cuda()
    # model = nn.DataParallel(model)
    charbert_model = initialise_charbert_model(experiment_version=None)
    charbert_model = charbert_model.to(device)
    model = model.to(device)
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    # for name,param in model.named_parameters():
    #     print(name)
    #     print(param)
    message = {"freeze": "Frozen parameters: ", "not_freeze": "Trainable parameters: "}
    if layers==[]:
        layers = [message[freeze_mode]] + [" "*27+"None"]
    else:
        layers = [message[freeze_mode]] + [" "*27+"Starts with "+i for i in layers]
    my_logger.info("\n".join(str(element) for element in layers))
    # Initialize Integrated Gradients
    # ig = IntegratedGradients(model)
    my_logger.info("Max epochs: %d", max_epochs)
    while (early_stopping==False) and (epoch<max_epochs):
        model_store = False
        epoch += 1
        # train_loss, train_wer_sum, train_cer_sum = 0, 0, 0
        train_loss = 0
        train_loss, train_wer_sum, train_cer_sum = 0, 0, 0
        val_loss, val_wer_sum, val_cer_sum = 0, 0, 0
        test_loss, test_wer_sum, test_cer_sum = 0, 0, 0
        train_samples, val_samples, test_samples = 0, 0, 0
        # Clear all contents in the text output file
        with open(text_output_path, "w") as file:
            pass
        # Training
        model.train()
        # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i, batch in enumerate(train_loader):
            # if i>3:
            #     break
            if epoch==1 and i==0:
                my_logger.info("Batch size = %d", batch["pixel_values"].size(0))
            optimizer.zero_grad()
                # if i>3:
                #     break
                # print("one batch")
            train_samples += batch["pixel_values"].size(0)
                # print("one batch")
                # train_samples += len(batch.items())
            for k,v in batch.items():
                    # batch["pixel_values"] = torch.rand((2, 3, 384, 384)).to(device)
                    # batch["attention_mask"] =  torch.randint(0, 1, (2, 512)).to(device)
                    # batch["label_emb"] = torch.rand((2, 512, 1024)).to(device)
                    # batch["labels"] = torch.randint(0, 100, (2, 512), dtype=torch.long).to(device)
                if k in ["pixel_values", "label_emb", "labels", "attention_mask", "test_decoder_input_ids", "start_ids", "end_ids", "char_input_ids", "input_ids"]:#, "charbert_token_labels", "charbert_char_labels"]:
                    batch[k] = v.to(device)
            charbert_output = charbert_model(char_input_ids=batch["char_input_ids"], start_ids=batch["start_ids"], end_ids=batch["end_ids"], input_ids=batch["input_ids"])
            charbert_token_label = charbert_output[0]
            charbert_char_label = charbert_output[2]
            # print("\nMODEL NAME: ", model_name)
            if model_name.startswith("trocr_charbert"):
                outputs, charbert_token_repr, charbert_char_repr = model(pixel_values=batch["pixel_values"], 
                                decoder_inputs_embeds=batch["label_emb"], 
                                decoder_attention_mask=batch["attention_mask"], labels=batch["labels"],
                                start_ids=batch["start_ids"], end_ids=batch["end_ids"])
            # outputs = model(pixel_values=batch["pixel_values"], 
            #                 decoder_inputs_embeds=batch["label_emb"], 
            #                 decoder_attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
            elif model_name=="trocr":
                # outputs = model(pixel_values=batch["pixel_values"], 
                #                 decoder_input_ids=batch["test_decoder_input_ids"], labels=batch["labels"])
                outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                loss = outputs.loss
            # pred_str = get_str(processor, outputs.logits)
            # wer, cer = get_wer_cer_per_batch(pred_str, batch["label_str"])
            # train_wer_sum += wer
            # train_cer_sum += cer
            # print("train output: ", outputs.logits.size())
                # print("train output type: ", outputs.logits.dtype)
                # print(outputs.logits)
                # print(batch["labels"])
            # print(outputs.logits.view(-1, outputs.logits.size(-1)).size())
            # print(batch["labels"].view(-1).size())
            # loss = outputs.loss
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
            loss_2 = 1 - F.cosine_similarity(charbert_token_repr, charbert_token_label).mean()
            loss_3 = 1 - F.cosine_similarity(charbert_char_repr, charbert_char_label).mean()
            # print(loss)
            # print(loss_2)
            # print(loss_3)
            # print(charbert_token_repr.view(-1, charbert_token_repr.size(-1)).size())
            # print(batch["input_ids"].view(-1).size())
            # loss_2 = criterion(charbert_token_repr.view(-1, charbert_token_repr.size(-1)), batch["input_ids"].view(-1))
            # loss_3 = criterion(charbert_char_repr.view(-1, charbert_char_repr.size(-1)), batch["char_input_ids"].view(-1))
            # loss = outputs.loss.sum()
            # print(loss)
            # print(type(loss))
                # print(loss)
                # print("train loss: ", loss)
            loss = loss/4 + loss_2 + loss_3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Check gradient
            for name, param in model.named_parameters():
            #     if name=="decoder.model.decoder.layers.8.fc1.weight":
                #     print(name, "gradient: ", param.grad_fn)
                #     print(param.grad)
                # print(f"{name}: requires_grad={param.requires_grad}")
                # if name.startswith("ffnn"):
                # if name.startswith("encoder"):
                #     print(param.grad)
                if param.grad is not None and (param.grad == 0).all():
                    print(name, " has zero gradient")
                #     # print(name, "   ", param.grad)
                # else:
                #     print(name, "   ", param.grad, "ok")
                # print(name, "   ", param.grad_fn)
                # if param.grad is not None:
                #     print(name, "ok")
            optimizer.step()
                # print("after: ", model.decoder.output_projection.weight)
            train_loss += loss.detach()
            pred_str = get_str(processor, outputs.logits)
            # if text_output:
            #     write_predictions_to_file(pred_str, batch["label_str"], text_output_path)
        train_wer_score = train_wer_sum/train_samples
        train_cer_score = train_cer_sum/train_samples
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # Validation
        with torch.no_grad():
            model.eval() 
            for i, batch in enumerate(val_loader):
                # if i>3:
                #     break
                val_samples += batch["pixel_values"].size(0)
                for k,v in batch.items():
                    if k in ["pixel_values", "label_emb", "labels", "attention_mask", "test_decoder_input_ids", "start_ids", "end_ids", "char_input_ids", "input_ids"]:
                        batch[k] = v.to(device)

                charbert_output = charbert_model(char_input_ids=batch["char_input_ids"], start_ids=batch["start_ids"], end_ids=batch["end_ids"], input_ids=batch["input_ids"])
                charbert_token_label = charbert_output[0]
                charbert_char_label = charbert_output[2]
                if model_name.startswith("trocr_charbert"):
                    outputs, charbert_token_repr, charbert_char_repr = model(pixel_values=batch["pixel_values"], decoder_inputs_embeds=batch["label_emb"], 
                                decoder_attention_mask=batch["attention_mask"], labels=batch["labels"],
                                start_ids=batch["start_ids"], end_ids=batch["end_ids"])
                # outputs = model(pixel_values=batch["pixel_values"], 
                #             decoder_inputs_embeds=batch["label_emb"], 
                #             decoder_attention_mask=batch["attention_mask"], labels=batch["labels"])
                    # generated_ids = model.generate(pixel_values=batch["pixel_values"])
                    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    # print(generated_text)
                    pred_str = get_str(processor, outputs.logits)
                    # pred_str = [filter_string(i) for i in pred_str]
                    # label_str = [filter_string(i) for i in batch["label_str"]]
                    wer, cer = get_wer_cer_per_batch(pred_str, batch["label_str"])
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                elif model_name=="trocr":
                    outputs = model(pixel_values=batch["pixel_values"], 
                                decoder_input_ids=batch["test_decoder_input_ids"], labels=batch["labels"])
                    generated_ids = model.generate(batch["pixel_values"])
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    # print("GENERATE TEXT: ", generated_text)
                    loss = outputs.loss
                    wer, cer = get_wer_cer_per_batch(generated_text, batch["label_str"])
                # loss = outputs.loss
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                loss_2 = 1 - F.cosine_similarity(charbert_token_repr, charbert_token_label).mean()
                loss_3 = 1 - F.cosine_similarity(charbert_char_repr, charbert_char_label).mean()
                # print(loss_1)
                # print(loss_2)
                # print(loss_3)
                loss = loss/4 + loss_2 + loss_3
                # print("val loss: ", loss)
                val_loss += loss.detach()
                # pred_str = get_str(processor, outputs.logits)
                # wer, cer = get_wer_cer_per_batch(generated_text, batch["label_str"])
                # if text_output:
                #     write_predictions_to_file(pred_str, batch["label_str"], text_output_path)
                val_wer_sum += wer
                val_cer_sum += cer
            # train_loss = train_loss/train_samples
            # val_loss = val_loss/val_samples
            val_wer_score = val_wer_sum/val_samples
            val_cer_score = val_cer_sum/val_samples
        # early stopping
        if val_loss <= lowest_loss:
            model_store = False
            lowest_loss = val_loss
            save_epoch = epoch
            # Save the fine-tuned model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            # torch.save(model.state_dict(), model_path)
            # my_logger.info("Model saved at epoch %d", epoch-1)
        # else:
        #     early_stopping = True
        #     my_logger.info("Early stopping at epoch %d.", epoch)
        #     my_logger.info("Model saved at epoch %d", epoch-1)
        with torch.no_grad():
            model.eval() 
            for i, batch in enumerate(test_loader):
                # if i>3:
                #     break
                test_samples += batch["pixel_values"].size(0)
                for k,v in batch.items():
                    if k in ["pixel_values", "label_emb", "labels", "attention_mask", "test_decoder_input_ids", "start_ids", "end_ids", "char_input_ids", "input_ids"]:
                        batch[k] = v.to(device)

                # charbert_output = charbert_model(char_input_ids=batch["char_input_ids"], start_ids=batch["start_ids"], end_ids=batch["end_ids"], input_ids=batch["input_ids"])
                # charbert_token_label = charbert_output[0]
                # charbert_char_label = charbert_output[2]
                if model_name.startswith("trocr_charbert"):
                    outputs, charbert_token_repr, charbert_char_repr = model(pixel_values=batch["pixel_values"], decoder_inputs_embeds=batch["label_emb"], 
                                decoder_attention_mask=batch["attention_mask"], labels=batch["labels"],
                                start_ids=batch["start_ids"], end_ids=batch["end_ids"])
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                    pred_str = get_str(processor, outputs.logits)
                    # pred_str = [filter_string(i) for i in pred_str]
                    # label_str = [filter_string(i) for i in batch["label_str"]]
                    wer, cer = get_wer_cer_per_batch(pred_str, batch["label_str"])
                    # wer, cer = get_wer_cer_per_batch(pred_str, batch["label_str"])
                # outputs = model(pixel_values=batch["pixel_values"], 
                #             decoder_inputs_embeds=batch["label_emb"], 
                #             decoder_attention_mask=batch["attention_mask"], labels=batch["labels"])
                elif model_name=="trocr":
                    outputs = model(pixel_values=batch["pixel_values"], 
                                decoder_input_ids=batch["test_decoder_input_ids"], labels=batch["labels"])
                    generated_ids = model.generate(batch["pixel_values"])
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    loss = outputs.loss
                    wer, cer = get_wer_cer_per_batch(generated_text, batch["label_str"])
                # loss_2 = 1 - F.cosine_similarity(charbert_token_repr, charbert_token_label).mean()
                # loss_3 = 1 - F.cosine_similarity(charbert_char_repr, charbert_char_label).mean()
                # print(loss_1)
                # print(loss_2)
                # print(loss_3)
                # loss = loss_1/4 + loss_2 + loss_3
                # print("val loss: ", loss)
                test_loss += loss.detach()
                # pred_str = get_str(processor, outputs.logits)
                if text_output and model_name=="trocr":
                    write_predictions_to_file(generated_text, batch["label_str"], text_output_path)
                if text_output and model_name.startswith("trocr_charbert"):
                    write_predictions_to_file(pred_str, batch["label_str"], text_output_path)
                test_wer_sum += wer
                test_cer_sum += cer
            # train_loss = train_loss/train_samples
            # val_loss = val_loss/val_samples
            test_wer_score = test_wer_sum/test_samples
            test_cer_score = test_cer_sum/test_samples
        if test_cer_score<lowest_cer:
            model_store = True
            lowest_cer = test_cer_score

            # This will copy the source file to the destination and overwrite it if it exists.
            shutil.copy2(text_output_path, text_output_path[:-4]+"_"+text_file+".txt")
        if epoch==1:
            my_logger.info(("{:>10} | {:>13} | {:>13} | {:>13} | {:>13} | {:>13} | {:>13}\n" + " "*27 + "{:>10} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} \tModel saved.").format("Epoch", "Train Loss", "Val Loss", "Val WER", "Val CER", "Test WER", "Test CER", 1, train_loss, val_loss, val_wer_score, val_cer_score, test_wer_score, test_cer_score))
        elif model_store:
            my_logger.info(("{:>10} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} \tModel saved.").format(epoch, train_loss, val_loss, val_wer_score, val_cer_score, test_wer_score, test_cer_score))
        elif not model_store:
            my_logger.info(("{:>10} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} | {:>13.4f} ").format(epoch, train_loss, val_loss, val_wer_score, val_cer_score, test_wer_score, test_cer_score))
        torch.cuda.empty_cache()
    # Save the figure
    # plt.savefig(os.path.join(RESULTS, "analysis.png"), bbox_inches="tight")
    # plt.close(fig)  # Close the figure to free memory
    my_logger.info("Model saved at epoch %d", save_epoch)
    my_logger.info("Training process completed. The logging file is stored in %s.", log_file_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Print the elapsed time
    my_logger.info("Elapsed time: {:.2f} seconds".format(elapsed_time))
    shutdown_logger(my_logger)
    return val_wer_score, val_cer_score