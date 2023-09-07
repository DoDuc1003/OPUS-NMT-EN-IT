from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_language, target_language, sequence_length) -> None:
        super().__init__()
        
        self.datset = dataset
        
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        
        self.source_language = source_language
        self.target_language = target_language
        

        self.sos_token = torch.tensor([tokenizer_target.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_target.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_target.token_to_id('[PAD]')], dtype=torch.int64)
        
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        return len(self.datset)
    
    def __getitem__(self, index: Any) -> Any:
        source_target_pair = self.datset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]
        
        encode_input_tokens = self.tokenizer_source.encode(source_text).ids
        decode_input_tokens = self.tokenizer_target.encode(target_text).ids 
        
        encode_number_padding_tokens = self.sequence_length - len(encode_input_tokens) - 2 # SOS and EOS
        decode_number_padding_tokens = self.sequence_length - len(decode_input_tokens) - 1 # EOS
        
        if encode_number_padding_tokens < 0 or decode_number_padding_tokens < 0:
            raise ValueError('sentence to long')
        
        # add SOS and EOS to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encode_number_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decode_number_padding_tokens, dtype=torch.int64)
            ]
        )
        
        # add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decode_number_padding_tokens, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert label.size(0) == self.sequence_length
        
        return {
            "encoder_input": encoder_input, #(sequence_length)
            "decoder_input": decoder_input, #(sequence_length)
            # unsqueeze(0) thêm chiều
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, sequence_length)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, sequence_length) & (1, sequence_length, sequence_length)
            "label": label, # (sequence_length)
            "source_text": source_text,
            "target_text": target_text
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask
        
        
        
        