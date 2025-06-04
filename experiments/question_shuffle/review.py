"""Implements a GUI which allows fast filtering of invalid entries in
a dataset.

Reads through the parquet, selects the top k entries sorted in order
of greatest reward, and displays them on the user interface. Includes
buttons to remove or keep this entry.

Default implementation not distinguish between different splits, and o-
nly considers the reward of the Skywork PRM.

If an entry is selected to be kept, it's tagged with "interesting":True
"""




import sys
import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib.pyplot as plt
from PIL import Image
import io




def sorting_func(entry):
    if entry["final_answer_correct"] == False:
        return -1
    return entry["Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"][-1] - entry["Skywork-o1-Open-PRM-Qwen-2.5-7B"][-1]

df = pd.read_parquet("attack.parquet")

df['_temp_sort_key'] = df.apply(sorting_func, axis=1)
df_sorted = df.sort_values(by='_temp_sort_key', ascending=False)
top = df_sorted.head(100)
assert top.iloc[-1]["final_answer_correct"] == True, "incorrect answers included"

print(top)
df.drop(columns=["_temp_sort_key"], inplace=True)

df["interesting"] = False

class FilterGUI:
    def __init__(self, root, full_df, top_subset):
        self.root = root
        self.full_df = full_df
        self.top_subset = top_subset.reset_index(drop=False)

        self.current_position = 0
        self.total_entries = len(self.top_subset)

        self.root.title("Fast Filtering GUI")
        self.root.geometry("1920x1080")

        # (3.1) A scrollable text widget to display the current row
        self.text_box = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Courier", 10)
        )
        self.text_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.text_box.config(state=tk.DISABLED)

        # (3.2) Frame for the buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.keep_btn = tk.Button(
            btn_frame,
            text="Keep ↦ Tag as Interesting",
            width=20,
            command=self.keep_entry,
        )
        self.keep_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.remove_btn = tk.Button(
            btn_frame,
            text="Remove ↦ Skip",
            width=20,
            command=self.remove_entry,
        )
        self.remove_btn.pack(side=tk.LEFT, padx=(5, 0))

        # (3.3) A status label (e.g., “3 / 100”)
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(fill=tk.X)

        # Display the very first entry
        self._display_current_entry()

def render_latex_to_image(latex: str, width: int, output_path: str = None) -> Image.Image:
    """
    Renders a LaTeX string to an image with the specified width (in pixels).

    Parameters:
    latex       (str):  Your LaTeX content (math‐mode only; if not wrapped in '$...$', the function will add it automatically).
    width       (int):  Desired image width in pixels.
    output_path (str, optional):  If provided, saves the resized image to this path.

    Returns:
    PIL.Image.Image:  A Pillow Image object containing the rendered LaTeX at the requested width.
    """
    # Ensure the string is in math mode
    if not (latex.startswith('$') and latex.endswith('$')):
        latex = f'${latex}$'

    # 1) Create a tiny Matplotlib figure purely to render the LaTeX
    fig = plt.figure(figsize=(0.01, 0.01), dpi=300)  # very small; bbox_inches='tight' will expand it
    fig.patch.set_alpha(0)  # transparent background

    # 2) Place the LaTeX text at position (0,0)
    fig.text(0, 0, latex, fontsize=20, usetex=True)

    # 3) Save to an in‐memory buffer as PNG with a tight bounding box
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format='png',
        bbox_inches='tight',
        pad_inches=0.0,
        transparent=True
    )
    plt.close(fig)
    buf.seek(0)

    # 4) Open that PNG buffer in PIL
    img = Image.open(buf)

    # 5) Compute new height so that we preserve aspect ratio
    original_width, original_height = img.size
    new_height = int((width / original_width) * original_height)

    # 6) Resize using a high‐quality filter
    img_resized = img.resize((width, new_height), Image.LANCZOS)

    # 7) If the user provided an output path, save it
    if output_path:
        img_resized.save(output_path)

    return img_resized
    

# Example LaTeX: a standard Gaussian integral
latex_expr = r"\int_{0}^{\infty} e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}"
requested_width = 400  # pixels

img = render_latex_to_image(latex_expr, requested_width, output_path="gaussian.png")
img.save("latex.jpg")