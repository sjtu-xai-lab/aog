import torch

# ==================
#       NLP
# ==================
def pad_baseline_nlp(model, text_field):
    device = next(model.parameters()).device
    pad_idx = text_field.vocab.stoi[text_field.pad_token]
    with torch.no_grad():
        tensor = torch.LongTensor([pad_idx]).to(device)
        baseline_value = model.get_emb(tensor)
    return baseline_value


def calc_word_emb_std(model, text_field):
    device = next(model.parameters()).device
    vocab_size = len(text_field.vocab)
    tensor = torch.LongTensor(list(range(vocab_size))).to(device)
    with torch.no_grad():
        embeddings = model.get_emb(tensor)
    return torch.std(embeddings, dim=0, keepdim=True)