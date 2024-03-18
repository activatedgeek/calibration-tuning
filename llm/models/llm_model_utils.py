DEFAULT_PAD_TOKEN = "[PAD]"


def resize_token_embeddings(tokenizer, model):
    extra_token_count = len(tokenizer) - model.get_input_embeddings().weight.data.size(
        0
    )
    if extra_token_count:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-extra_token_count:] = input_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)

        output_embeddings = model.get_output_embeddings().weight.data

        output_embeddings[-extra_token_count:] = output_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)
