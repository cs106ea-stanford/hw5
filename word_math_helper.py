### INSTALL "Approximate Nearest Neighbor Oh Yeah" Library
from annoy import AnnoyIndex

import numpy as np
import torch

# Switched from using GloVe from TorchText because I had compatibility issues with PyTorch and TorchVision
# GloveLoader code care of ChatGPT-4o is meant to be swap for TorchText version

class GloveLoader:
    def __init__(self, file_path):
        self.stoi = {}  # String-to-Index mapping
        self.itos = []  # Index-to-String mapping
        vectors = []  # Word vectors

        # Load GloVe file
        with open(file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)

                self.stoi[word] = i
                self.itos.append(word)
                vectors.append(vector)

        # Convert to a single matrix (torch.Tensor) for efficient indexing
        self.vectors = torch.tensor(np.array(vectors))  # Shape: (num_words, dim)

    def __getitem__(self, word):
        """Retrieve the word vector (as a torch.Tensor)."""
        if word in self.stoi:
            return self.vectors[self.stoi[word]]
        return None  # Handle out-of-vocab words

    def get_vector_by_index(self, index):
        """Retrieve a vector by index (like TorchText)."""
        return self.vectors[index]

    def get_word_by_index(self, index):
        """Retrieve a word by index (like TorchText)."""
        return self.itos[index] if 0 <= index < len(self.itos) else None

# Load the GloVe embeddings
glove50 = GloveLoader("glove.6B.50d.txt")

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class EmbedSet:
    vocab: GloveLoader
    neighbor_index: AnnoyIndex

    def __post_init__(self)->None:
        for word, idx in self.vocab.stoi.items():
            vector = self.vocab[word]
            self.neighbor_index.add_item(idx,vector.numpy())
        self.neighbor_index.build(10)

    def word_math(self, phrase: str,n: int)->List[str]:
        parent_match_node = parse_phrase(phrase, self)
        return parent_match_node.execute_return_nearest(n)

    def __getitem__(self,key: str)->torch.tensor:
        return self.vocab[key]

    def nearest_neighbors(self,embed_vector: torch.tensor,n:int=5)->list[str]:
        nearest_indices = self.neighbor_index.get_nns_by_vector(embed_vector.numpy(),n)
        return [self.vocab.itos[index] for index in nearest_indices]

vs: EmbedSet = None

# Replace set_glove_embed with:
def set_glove_embed(glove_loader):
    global vs
    vs = EmbedSet(
        glove_loader,
        AnnoyIndex(f=50, metric='angular')
    )

set_glove_embed(glove50)

### WORD MATH

class MathNode():
    def __init__(self, vocab_set: EmbedSet):
        self.vocab_set = vocab_set

    # abstract method
    def execute(self)->torch.tensor:
        pass

    def execute_return_nearest(self, n:int=5):
        return self.vocab_set.nearest_neighbors(self.execute(),n)

class ConstantNode(MathNode):
    def __init__(self, phrase: str, vocab_set: EmbedSet):
        super().__init__(vocab_set)
        self.value = phrase.strip().lower()
        self.tensor = self.vocab_set.vocab[self.value]

    def execute(self)->torch.tensor:
        return self.tensor

class AddNode(MathNode):
    def __init__(self,left_phrase,right_phrase, vocab_set: EmbedSet):
        super().__init__(vocab_set)
        self.left = parse_phrase(left_phrase, vocab_set)
        self.right = parse_phrase(right_phrase, vocab_set)

    def execute(self)->torch.tensor:
        return self.left.execute() + self.right.execute()

class SubtractNode(MathNode):
    def __init__(self,left_phrase,right_phrase, vocab_set: EmbedSet):
        super().__init__(vocab_set)
        self.left = parse_phrase(left_phrase, vocab_set)
        self.right = parse_phrase(right_phrase, vocab_set)

    def execute(self)->torch.tensor:
        return self.left.execute() - self.right.execute()

def parse_phrase(phrase: str, vocab_set: EmbedSet)->MathNode:
    plus_index = phrase.find('+')
    minus_index = phrase.find('-')

    if plus_index == -1 and minus_index == -1:
        return ConstantNode(phrase, vocab_set)

    # print(f"plus_index={plus_index}")
    # print(f"minus_index={minus_index}")

    if plus_index == -1:
        # print(f"Sub Node with ({phrase[:minus_index]}) and ({phrase[minus_index+1:]})")
        return SubtractNode(phrase[:minus_index],phrase[minus_index+1:], vocab_set)
    else:
        # print(f"Add Node with ({phrase[:plus_index]}) and ({phrase[plus_index+1:]})")
        return AddNode(phrase[:plus_index],phrase[plus_index+1:], vocab_set)

def eval_phrase(phrase: str, vocab_set: EmbedSet, n: int=10)->None:
    output = vocab_set.word_math(phrase, n)
    print(f"{phrase} =\n\t {output}")

# solves the analogy x is to y as a is to b, given x, y, and b returns a
# provide with strings x, y, b, an EmbedSet, and the number of neighbors desired
# returns a list of strings corresponding to the desired number of neighbors
def analogy_standard(x: str, y: str, b: str, vocab_set: EmbedSet, n: int=10)->List[str]:
    return vocab_set.word_math(f"{x} - {y} + {b}", n)

### GUI

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, HTML, Label
from IPython.display import clear_output

from typing import List

layout_for_entry_text = Layout(width="120px")

a_text = widgets.Text(layout=layout_for_entry_text)
a_to_b_html = HTML("&nbsp;is to&nbsp;")
b_text = widgets.Text(layout=layout_for_entry_text)
b_to_x_html = HTML("&nbsp;&nbsp;&nbsp;as&nbsp;&nbsp;&nbsp;")
x_blank = HTML("_____________")
x_to_y_html = HTML("&nbsp;is to&nbsp;")
y_text = widgets.Text(layout=layout_for_entry_text)
solve_btn = widgets.Button(description="Solve Analogy", button_style='info')

analogy_entry = HBox([a_text, a_to_b_html, b_text, b_to_x_html,
                     x_blank, x_to_y_html, y_text, solve_btn])

preset_html = HTML("<b>Choose a preset value or enter your own words:</b>")
preset_entries = [
    ["London", "England", "Japan"],
    ["Airplane", "Airport", "Port"],
    ["Doctor", "Man", "Woman"],
    ["Doctor", "Woman", "Man"]
]

a_text.value = preset_entries[0][0]
b_text.value = preset_entries[0][1]
y_text.value = preset_entries[0][2]

options = [f"{entry[0]} is to {entry[1]} as __________ is to {entry[2]}" for entry in preset_entries]
options.append("Custom")

preset_dropdown = widgets.Dropdown(options=options,
                                   layout=Layout(width="350px"))

def get_selected_index(dropdown_widget):
    return dropdown_widget.options.index(dropdown_widget.value)

def use_preset(_):
    curr_index = get_selected_index(preset_dropdown)

    if curr_index < len(preset_entries):
        curr_entry = preset_entries[curr_index]
        a_text.value = curr_entry[0]
        b_text.value = curr_entry[1]
        y_text.value = curr_entry[2]
    else:
        a_text.value = ""
        b_text.value = ""
        y_text.value = ""

preset_dropdown.observe(use_preset, names='value')

preset_entry = VBox([preset_html, preset_dropdown],
                        layout=Layout(margin="0 0 20px 0"))

analogy_html_output = HTML(layout=Layout(height="300px"))

def output_list(lst: List[str], output_html: widgets.HTML):
    html_str = ""
    for item in lst:
        html_str += item + "<br>"
    output_html.value = html_str

def output_warning(error_words: List[str], output_html: widgets.HTML):
    html_str = "<b>Error:</b> The following words are not in the database<br>"
    for item in error_words:
        html_str += item + "<br>"
    output_html.value = html_str

def check_word(word:str, vocab_set):
    word = word.strip().lower()
    try:
        vocab_set[word]
        return True
    except KeyError:
        return False

def solve_analogy_and_output(_):
    error = False
    error_words = []
    if not check_word(a_text.value,vs):
        error = True
        error_words.append(a_text.value,vs)
    if not check_word(b_text.value,vs):
        error = True
        error_words.append(b_text.value,vs)
    if not check_word(y_text.value,vs):
        error = True
        error_words.append(y_text.value,vs)

    if error:
        output_warning(error_words, analogy_html_output)
        return

    analogy_list = analogy_standard(a_text.value, b_text.value, y_text.value, vs)

    output_list(analogy_list, analogy_html_output)

solve_btn.on_click(solve_analogy_and_output)

def display_analogy_ui():
    display(VBox([preset_entry, analogy_entry, analogy_html_output]))

