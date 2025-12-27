"""
Copied from  https://github.com/bo-miao/SgMg
Modified to use MiniLM for efficiency
"""
import torch
from torch import nn, Tensor
from transformers import AutoModel
from transformers import RobertaModel  # Original RoBERTa (commented out)
from models.uniphd.text_encoder.tokenizer import (
    MiniLMTokenizer, 
    DistilBERTTokenizer, 
    TinyBERTTokenizer, 
    TiTeLATETokenizer,
    ALBERTTokenizer,
    MobileBERTTokenizer,
    RobertaTokenizer  # Original (commented out)
)

import warnings
warnings.filterwarnings("ignore")


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.text_backbone_name = "ALBERT"  # Options: MiniLM, TinyBERT, ALBERT, MobileBERT, DistilBERT, TiTeLATE, Roberta'
        self.token_size = 32
        
        if self.text_backbone_name == "MiniLM":
            # Using sentence-transformers/all-MiniLM-L6-v2 (23M params, 384 dim)
            self.text_backbone = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizer = MiniLMTokenizer()
            self.feat_dim = 384  # MiniLM output dimension
            self.use_mean_pooling = True
            
        elif self.text_backbone_name == "DistilBERT":
            # Using distilbert/distilbert-base-uncased (66M params, 768 dim)
            self.text_backbone = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
            self.tokenizer = DistilBERTTokenizer()
            self.feat_dim = 768  # DistilBERT output dimension
            self.use_mean_pooling = True  # DistilBERT uses mean pooling (no pooler)
            
        elif self.text_backbone_name == "TinyBERT":
            # Using huawei-noah/TinyBERT_General_4L_312D (14.5M params, 312 dim)
            self.text_backbone = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
            self.tokenizer = TinyBERTTokenizer()
            self.feat_dim = 312  # TinyBERT output dimension
            self.use_mean_pooling = False  # TinyBERT has pooler
            
        elif self.text_backbone_name == "ALBERT":
            # Using albert/albert-base-v2 (11.8M params, 768 dim with 128 embedding dim)
            self.text_backbone = AutoModel.from_pretrained("albert/albert-base-v2")
            self.tokenizer = ALBERTTokenizer()
            self.feat_dim = 768  # ALBERT hidden dimension
            self.use_mean_pooling = False  # ALBERT has pooler
            
        elif self.text_backbone_name == "MobileBERT":
            # Using google/mobilebert-uncased (25M params, 512 dim)
            self.text_backbone = AutoModel.from_pretrained("google/mobilebert-uncased")
            self.tokenizer = MobileBERTTokenizer()
            self.feat_dim = 512  # MobileBERT output dimension
            self.use_mean_pooling = False  # MobileBERT has pooler
            
        elif self.text_backbone_name == "TiTeLATE":
            # Using webis/tite-2-late-msmarco (lightweight BERT variant, 768 dim)
            self.text_backbone = AutoModel.from_pretrained("webis/tite-2-late-msmarco")
            self.tokenizer = TiTeLATETokenizer()
            self.feat_dim = 768  # TiTe-LATE output dimension
            self.use_mean_pooling = False  # TiTe-LATE has pooler
            
        elif self.text_backbone_name == "Roberta":  # Original RoBERTa (commented out)
            self.text_backbone = RobertaModel.from_pretrained("roberta-base")
            # self.text_backbone.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
            self.tokenizer = RobertaTokenizer()
            self.feat_dim = 768
            self.use_mean_pooling = False
        else:
            assert False, f'error: Text Encoder "{self.text_backbone_name}" is not supported. Supported: MiniLM, TinyBERT, ALBERT, MobileBERT, DistilBERT, TiTeLATE, Roberta'

        self.freeze_text_encoder = args.freeze_text_encoder
        if self.freeze_text_encoder:
            for p in self.text_backbone.parameters():
                p.requires_grad_(False)
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)
        print("Use {} as text encoder. Freeze: {}".format(self.text_backbone_name, self.freeze_text_encoder))

        self.target_len = None

    def forward(self, texts, device):
        if self.freeze_text_encoder:
            with torch.no_grad():
                tokenized_queries = self.tokenizer(texts).to(device)
                encoded_text = self.text_backbone(**tokenized_queries)
                text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()
                text_features = encoded_text.last_hidden_state
                
                # Choose pooling strategy based on model type
                if self.use_mean_pooling:
                    # Models without pooler: MiniLM, DistilBERT
                    text_sentence_features = self._mean_pooling(encoded_text.last_hidden_state, 
                                                                 tokenized_queries.attention_mask)
                else:
                    # Models with pooler: TinyBERT, TiTe-LATE, RoBERTa
                    text_sentence_features = encoded_text.pooler_output
        else:
            tokenized_queries = self.tokenizer(texts).to(device)
            encoded_text = self.text_backbone(**tokenized_queries)
            text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()
            text_features = encoded_text.last_hidden_state
            
            # Choose pooling strategy based on model type
            if self.use_mean_pooling:
                # Models without pooler: MiniLM, DistilBERT
                text_sentence_features = self._mean_pooling(encoded_text.last_hidden_state, 
                                                             tokenized_queries.attention_mask)
            else:
                # Models with pooler: TinyBERT, TiTe-LATE, RoBERTa
                text_sentence_features = encoded_text.pooler_output

        return text_features, text_sentence_features, text_pad_mask
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling for sentence embeddings (MiniLM standard approach)"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

