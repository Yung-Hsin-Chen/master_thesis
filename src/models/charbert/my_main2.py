from modeling.modeling_charbert import CharBertModel, CharBertForMaskedLM
from modeling.modeling_roberta import RobertaModel, RobertaForMaskedLM
from run_lm_finetuning import load_and_cache_examples, TextDataset, set_seed, evaluate, train
from modeling.configuration_bert import BertConfig
from modeling.configuration_roberta import RobertaConfig
from transformers import BertTokenizer, RobertaTokenizer, WEIGHTS_NAME
import argparse
import os
import torch
import logging
from ..src.processor.data_loader import CustomDataset

MODEL_CLASSES = {
    'bert': (BertConfig, CharBertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

logger = logging.getLogger(__name__)

class MainArgs:
    def __init__(self):
        # (str) The input training data file (a text file).
        self.train_data_file = "./train_data.txt"
        # (str) The output directory where the model predictions and checkpoints will be written.
        self.output_dir = "./results/"
        # (str) An optional input evaluation data file to evaluate the perplexity on (a text file).
        self.eval_data_file = "./eval_data.txt"
        # (str) Char vocab for charBert.
        self.char_vocab = "./data/dict/roberta_char_vocab"
        # (str) Term vocab for charBert.
        self.term_vocab = "./data/dict/term_vocab"
        # (str) The model architecture to be fine-tuned.
        self.model_type = "roberta"
        # (str) The model checkpoint for weights initialization.
        self.model_name_or_path = "roberta-large"
        # (bool) Train with masked-language modeling loss instead of language modeling.
        self.mlm = True
        # (float) Ratio of tokens to mask for masked language modeling loss.
        self.mlm_probability = 0.10
        # (float) Ratio of tokens to mask for masked language modeling loss.
        self.adv_probability = 0.10
        # (str) Optional pretrained config name or path if not the same as model_name_or_path.
        self.config_name = ""
        # (str) Training data version for cached file.
        self.data_version = ""
        # (str) Optional pretrained tokenizer name or path if not the same as model_name_or_path.
        self.tokenizer_name = ""
        # (str) Optional directory to store the pre-trained models downloaded from s3 (instead of the default one).
        self.cache_dir = ""
        # (int) Optional input sequence length after tokenization.
        self.block_size = -1
        # (bool) Whether to run training.
        self.do_train = True
        # (bool) Whether to run eval on the dev set.
        self.do_eval = True
        # (bool) Whether to output the debug information.
        self.output_debug = False
        # (bool) Run evaluation during training at each logging step.
        self.evaluate_during_training = True
        # (bool) Set this flag if you are using an uncased model.
        self.do_lower_case = False
        # (int) Max number of char for each word.
        self.char_maxlen_for_word = 6
        # (int) Batch size per GPU/CPU for training.
        self.per_gpu_train_batch_size = 2
        # (int) Batch size per GPU/CPU for evaluation.
        self.per_gpu_eval_batch_size = 2
        # (int) Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_accumulation_steps = 1
        # (float) The initial learning rate for Adam.
        self.learning_rate = 5e-5
        # (float) Weight decay if we apply some.
        self.weight_decay = 0.0
        # (float) Epsilon for Adam optimizer.
        self.adam_epsilon = 1e-8
        # (float) Max gradient norm.
        self.max_grad_norm = 1.0
        # (float) Total number of training epochs to perform.
        self.num_train_epochs = 2.0
        # (int) If > 0: set total number of training steps to perform. Override num_train_epochs.
        self.max_steps = -1
        # (int) Linear warmup over warmup_steps.
        self.warmup_steps = 0
        # (int) Log every X updates steps.
        self.logging_steps = 50
        # (int) Save checkpoint every X updates steps.
        self.save_steps = 50
        # (int) Number of lines when read the input data each time.
        self.input_nraws = 10000
        # (int) Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default.
        self.save_total_limit = None
        # (bool) Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number.
        self.eval_all_checkpoints = False
        # (bool) Avoid using CUDA when available.
        self.no_cuda = False
        # (bool) Overwrite the content of the output directory.
        self.overwrite_output_dir = False
        # (bool) Overwrite the cached training and evaluation sets.
        self.overwrite_cache = False
        # (int) Random seed for initialization.
        self.seed = 42
        # (bool) Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.
        self.fp16 = False
        # (str) For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
        self.fp16_opt_level = 'O1'
        # (int) For distributed training: local_rank.
        self.local_rank = -1
        # (str) For distant debugging.
        self.server_ip = ''
        # (str) For distant debugging.
        self.server_port = ''
args = MainArgs()

    


def main():

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=False,
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

if __name__=="__main__":
    main()
