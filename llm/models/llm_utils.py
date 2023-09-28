import logging
from peft import PeftModel, PeftConfig, get_peft_model

from ..utils.trainer import get_last_checkpoint_path


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_BOA_TOKEN = "<a>"
DEFAULT_EOA_TOKEN = "</a>"


def get_special_tokens(tokenizer):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict


def get_custom_tokens():
    return [DEFAULT_BOA_TOKEN, DEFAULT_EOA_TOKEN]


def load_peft_model_from_pretrained(model, peft_dir=None, query_peft_dir=None):
    if peft_dir is not None:
        peft_dir = get_last_checkpoint_path(peft_dir)

        model = PeftModel.from_pretrained(model, peft_dir)

        logging.info(f"Loaded PEFT checkpoint from '{peft_dir}'")

    if query_peft_dir is not None:
        query_peft_dir = get_last_checkpoint_path(query_peft_dir)

        ## NOTE: Hack to add a zero "default" adapter, for uniform downstream usage.
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, PeftConfig.from_pretrained(query_peft_dir))
            for n, p in model.named_parameters():
                if "lora" in n:
                    p.data.fill_(0.0)

        model.load_adapter(query_peft_dir, adapter_name="query")

        logging.info(f"Loaded PEFT query checkpoint from '{query_peft_dir}'")

    return model
