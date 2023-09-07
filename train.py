import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
import os
from tqdm import tqdm
import torchmetrics

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config


from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

from torch.utils.tensorboard import SummaryWriter


from pathlib import Path

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequery=1)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config["source"]}-{config["target"]}', split='train')
    
    # build tokenizer
    tokenizer_source = get_or_build_tokenizer(config=config, dataset=dataset_raw, language=config['source'])
    tokenizer_target = get_or_build_tokenizer(config=config, dataset=dataset_raw, language=config['target'])
    
    train_datset_size = int(0.9 * len(dataset_raw))
    validate_datset_size = len(dataset_raw) - train_datset_size
    
    train_datset_raw, validate_datset_raw = random_split(dataset=dataset_raw, lengths= [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    
    train_datset = BilingualDataset(train_datset_raw, tokenizer_source, tokenizer_target, config["source"], config["target"], config["sequence_length"])
    validate_datset = BilingualDataset(validate_datset_raw, tokenizer_source, tokenizer_target, config["source"], config["target"], config["sequence_length"])
    
    max_length_source = 0
    max_length_target = 0
    
    for item in dataset_raw:
        source_ids = tokenizer_source.encode(item["translation"][config['source']])
        target_ids = tokenizer_target.encode(item["translation"][config['target']])
        
        max_length_source = max(max_length_source, len(source_ids))
        max_length_target = max(max_length_target, len(target_ids))
    
    print(f"max length of source sentence : {max_length_source}")
    print(f"max length of target sentence : {max_length_target}")
    
    train_dataloader = DataLoader(dataset=train_datset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(dataset=validate_datset, batch_size=config["batch_size"], shuffle=False)
    
    return train_dataloader, valid_dataloader, tokenizer_source, tokenizer_target

def get_model(config, len_vocab_source, len_vocab_target):
    model = build_transformer(len_vocab_source, len_vocab_target, config["sequence_length"], config["sequence_length"], config["d_model"])
    return model

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        print(f"global step {global_step} cer")
        print(cer)

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        print(f"global step {global_step} wer")
        print(wer)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        print(f"global step {global_step} bleu score")
        print(bleu)

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, valid_dataloader, tokenizer_source, tokenizer_target = get_dataset(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device=device)
    
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config["preload"]:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device=device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_function(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        # save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
    
    
        
        
        

def main():
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

if __name__ == '__main__':
    main()