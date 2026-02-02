from flask import Flask, request, render_template
import os
import pickle
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

# --------------------
# App + Device
# --------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")


# --------------------
# Model (must match notebook)
# --------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, hidden):
        emb = self.dropout(self.embedding(src))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        pred = self.fc(out)
        return pred, hidden

    def init_hidden(self, batch_size, device, num_layers, hid_dim):
        h = torch.zeros(num_layers, batch_size, hid_dim).to(device)
        c = torch.zeros(num_layers, batch_size, hid_dim).to(device)
        return (h, c)


# --------------------
# Load vocab + weights
# --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "..", "model", "vocab_lm.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "model", "lstm_lm.pt")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

VOCAB_SIZE = len(vocab["itos"])
EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

model = LSTMLanguageModel(VOCAB_SIZE, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()


# --------------------
# Generation
# --------------------
@torch.no_grad()
def generate(prompt, max_new_tokens=40, temperature=0.9):
    model.eval()

    tokens = tokenizer(prompt)
    if len(tokens) == 0:
        tokens = ["<unk>"]

    ids = [vocab["stoi"].get(t, 0) for t in tokens]  # 0 = <unk>
    hidden = model.init_hidden(1, device, NUM_LAYERS, HID_DIM)

    # warm-up with prompt
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    _, hidden = model(x, hidden)

    out_ids = ids[:]
    last_id = ids[-1]

    for _ in range(max_new_tokens):
        x = torch.tensor([[last_id]], dtype=torch.long).to(device)
        pred, hidden = model(x, hidden)

        logits = pred[0, -1] / max(float(temperature), 1e-8)
        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, 1).item()
        out_ids.append(next_id)
        last_id = next_id

    # decode nicely (hide <eos>, show [UNK])
    words = [vocab["itos"][i] for i in out_ids]
    words = [w for w in words if w != "<eos>"]
    words = ["[UNK]" if w == "<unk>" else w for w in words]
    return " ".join(words)


# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    output = ""
    max_new = 40
    temperature = 0.9

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        max_new = int(request.form.get("max_new", 40))
        temperature = float(request.form.get("temperature", 0.9))
        output = generate(prompt, max_new, temperature)

    return render_template(
        "index.html",
        prompt=prompt,
        output=output,
        max_new=max_new,
        temperature=temperature,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
