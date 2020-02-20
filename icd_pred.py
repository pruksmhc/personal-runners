import pandas as pd

"""
Inspired by: https://github.com/sobamchan/pytorch-lightning-transformers/blob/master/mrpc.py
Inspired by https://github.com/suamin/multilabel-classification-bert-icd10/blob/master/load_data.py
"""
from IgniteF1 import IgniteMacroF1
 
from typing import Dict
from collections import OrderedDict
from functools import partial
import functools
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertForSequenceClassification,BertPreTrainedModel, BertTokenizer, AdamW,BertConfig, BertModel,PreTrainedModel
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLMTokenizer,
)
import numpy as np
import pickle
BertLayerNorm = torch.nn.LayerNorm
import torch.optim.lr_scheduler as lr_scheduler

import logging as log
@functools.lru_cache(maxsize=8, typed=False)
def get_tokenizer(tokenizer_name):
    log.info(f"\tLoading Tokenizer {tokenizer_name}")
    if tokenizer_name.startswith("bert-"):
        do_lower_case = tokenizer_name.endswith("cased")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("roberta-"):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("albert-"):
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("xlnet-"):
        do_lower_case = tokenizer_name.endswith("cased")
        tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("openai-gpt"):
        tokenizer = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("transfo-xl-"):
        # TransformerXL is trained on data pretokenized with MosesTokenizer
        tokenizer = MosesTokenizer()
    elif tokenizer_name.startswith("xlm-"):
        tokenizer = XLMTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "MosesTokenizer":
        tokenizer = MosesTokenizer()
    elif tokenizer_name == "SplitChars":
        tokenizer = SplitCharsTokenizer()
    elif tokenizer_name == "":
        tokenizer = SpaceTokenizer()
    else:
        tokenizer = None
    return tokenizer

BATCH_SIZE = 8
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    "bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def init_bert_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



MAX_LEN = 512
NUM_LABELS = 2
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, num_labels=2, hidden_dropout_prob=0.2, hidden_size=768, config=BertConfig.from_pretrained('bert-base-cased')):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.activation = torch.nn.Sigmoid()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, class_weights=None):
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        _, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask) 
        # to change, use the hidden states not the pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.activation(self.classifier(pooled_output))

        import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

# you could do 100, 100, 100,, 100.

def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:
    # Given two sentences, x["string1"] and x["string2"], this function returns BERT ready inputs.
    inputs = tokenizer.encode_plus(
        x["text"],
        add_special_tokens=True,
        max_length=MAX_LEN,
        )

    # First `input_ids` is a sequence of id-type representation of input string.
    # Second `token_type_ids` is sequence identifier to show model the span of "string1" and "string2" individually.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = MAX_LEN - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Super simple validation.
    assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

    # Convert them into PyTorch format.
    label = torch.tensor(x["labels"])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    # DONE!
    return {
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
            }

def tokenize_and_truncate(tokenizer_name, sent, max_seq_len):
    """Truncate and tokenize a sentence or paragraph."""
    max_seq_len -= 2  # For boundary tokens.
    tokenizer = get_tokenizer(tokenizer_name)

    if isinstance(sent, str):
        return tokenizer.tokenize(sent)[:max_seq_len]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return sent[:max_seq_len]

def process_discharge_summary(tokenizer_name,  text, max_seq_len):
	text = text.split(" ")
	return tokenize_and_truncate(tokenizer_name, text, max_seq_len)

def create_one_embedding(indices, num_labels):
    res = torch.zeros((1, num_labels))
    res[:,indices] = 1
    return res
def process_labels(labels, label_map):
    # you ahve to index the labes
    res_labels = [[label_map[x] for x in y if len(x) != 1] for y in labels]
    _, counts = np.unique([x for y in res_labels for x in y], return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    res_labels = [ create_one_embedding(torch.Tensor(x).long(), len(label_map)+1) for x in res_labels]
    return res_labels, weights

def get_X_y_ids(input_file, tokenizer_name, label_map, max_seq_len=256,
                as_heirarchy=False, max_sents_in_doc=10, max_words_in_sent=40,
                is_test=False):
    import os
    if os.path.exists("%s.pkl" % input_file):
        examples = pickle.load(open("%s.pkl" %input_file , "rb"))
        weights = pickle.load(open("%s.weights.pkl" % input_file, "rb"))
        return examples, weights
    file_contents = pd.read_csv(input_file)
    # https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/5
    text = file_contents["TEXT"].apply(lambda x: process_discharge_summary(tokenizer_name, x, max_seq_len))
    labels = file_contents["LABELS"].apply(lambda x: x.split(";") if isinstance(x, str) else str(x)).tolist()
    labels, weights = process_labels(labels, label_map)
    examples = [{"text":x[0], "labels":x[1]} for x in list(zip(text, labels)) if label_map["nan"] not in x[1]]
    examples = [x for x in examples if len(x["labels"]) > 0]
    pickle.dump(examples, open("%s.pkl" %input_file , 'wb'))
    pickle.dump(weights, open("%s.weights.pkl" % input_file, "wb"))
    return examples, weights

def get_dataloader():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    train = pd.read_csv("disch_train_split.csv")
    val = pd.read_csv("disch_dev_split.csv")
    test =  pd.read_csv("disch_test_split.csv")
    label_test_set = test["LABELS"].unique()
    label_train_set = train["LABELS"].unique()
    val_train_set = val["LABELS"].unique()
    label_test_set = [y.split(";") if isinstance(y, str) else [str(y)] for y in label_test_set]
    label_train_set = [y.split(";") if isinstance(y, str) else [str(y)] for y in label_train_set]
    val_train_set = [y.split(";") if isinstance(y, str) else [str(y)] for y in val_train_set]
    label_train_set = [x for y in label_train_set for x in y]
    label_test_set = [x for y in label_test_set for x in y]
    val_train_set = [x for y in val_train_set for x in y]
    total_labels = list(set(label_train_set + val_train_set + label_test_set))
    label_map = {total_labels[i]:i for i in range(len(total_labels))}
    train, t_weights = get_X_y_ids("disch_train_split.csv","bert-base-cased", label_map)
    val, _ = get_X_y_ids("disch_dev_split.csv", "bert-base-cased",label_map)
    test, _ = get_X_y_ids("disch_test_split.csv", "bert-base-cased", label_map)
    preprocessor = partial(preprocess, tokenizer)
    batch_size=BATCH_SIZE
    train_dataloader = DataLoader(
            list(map(preprocessor, train)),
            sampler=RandomSampler(train),
            batch_size=batch_size
            )
    val_dataloader = DataLoader(
            list(map(preprocessor,val)),
            sampler=SequentialSampler(val),
            batch_size=batch_size
            )
    test_dataloader = DataLoader(
            list(map(preprocessor,test)),
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader, len(label_map) + 1, t_weights

class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        self.class_threshold = 0.5
        train_dataloader, val_dataloader, test_dataloader, num_labels, weights = get_dataloader()
        self.class_weights = weights
        model = BertForMultiLabelSequenceClassification( num_labels=num_labels)
        self.model = model
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                )
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=self.class_weights
                )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
            })

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=self.class_weights
                )
        preds_hot = (logits >= self.class_threshold)
        labels = labels.squeeze(1) 
        correct_count = torch.sum(labels == preds_hot.float())

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)
        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": BATCH_SIZE, 
            "preds_hot": preds_hot.float(), 
            "labels": labels
            })
        return output

    def validation_end(self, outputs):
        val_f1 = IgniteMacroF1()
        for i in range(len(outputs)):
            val_f1(outputs[i]["preds_hot"], outputs[i]["labels"])
        val_metric_f1 = val_f1.get_metric()
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_metric_f1
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=self.class_weights
                )
        preds_hot = (logits >= self.class_threshold)
        labels = labels.squeeze(1)
        correct_count = torch.sum(labels == preds_hot.float())
        if self.on_gpu:
            correct_count = correct_count.cuda()

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": BATCH_SIZE,
            "preds_hot": preds_hot.float(),
            "labels": labels
            })

        return output

    def test_end(self, outputs):
        val_f1 = IgniteMacroF1()
        for i in range(len(outputs)):
            val_f1(outputs[i]["preds_hot"], outputs[i]["labels"])
        val_metric_f1 = val_f1.get_metric()
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": val_metric_f1    
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode="min"
            )

    trainer = pl.Trainer(
            gpus=1,
            early_stop_callback=early_stop_callback,
            )

    model = Model()

    trainer.fit(model)
    trainer.test()



