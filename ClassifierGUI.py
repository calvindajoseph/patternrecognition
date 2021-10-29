"""
A simple tkinter GUI for model demonstration.
"""

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Import ModelClassifier
from ModelClasses import ModelClassifier

# Import tkinter
import tkinter as tk

# Set the classifier
# May take some time, especially if loaded to CPU
classifier = ModelClassifier()

# Set main window
window = tk.Tk()
window.title("Sequence Classifier")

def predict():
    """
    The model prediction.
    
    Full workings are simpler to explain in ClassifierExample.py
    """
    first = str(first_sentence.get())
    second = str(second_sentence.get())
    prediction_str = classifier.print_prediction(first, second)
    txt_logs.delete('1.0', tk.END)
    txt_logs.insert(tk.INSERT, prediction_str)

# Set tkinter variables for first and second sentence
first_sentence = tk.StringVar()
second_sentence = tk.StringVar()

# Set label for the first sentence
lbl_first_sentence = tk.Label(master=window, text="First Sentence:")
lbl_first_sentence.grid(
    row=0,
    column=0,
    sticky="w",
    padx=5,
    pady=5)

# Set label for the second sentence
lbl_second_sentence = tk.Label(master=window, text="Second Sentence:")
lbl_second_sentence.grid(
    row=1,
    column=0,
    sticky="w",
    padx=5,
    pady=5)

# Set entry space for the first sentence
entry_first_sentence = tk.Entry(master=window, width=50, textvariable=first_sentence)
entry_first_sentence.grid(
    row=0,
    column=1,
    sticky="nswe",
    padx=5,
    pady=5)

# Set entry space for the second sentence
entry_second_sentence = tk.Entry(master=window, width=50, textvariable=second_sentence)
entry_second_sentence.grid(
    row=1,
    column=1,
    sticky="nswe",
    padx=5,
    pady=5)

# Set the button to run the classifier
btn_classify = tk.Button(
    master=window,
    text="Classify",
    command=predict)

btn_classify.grid(
    row=2,
    column=0,
    columnspan=2,
    sticky="nswe",
    padx=5,
    pady=5)

# Text that contains the results
txt_logs = tk.Text(
    master=window,
    height=15,
    width=50,
    fg="white",
    bg="black")

txt_logs.grid(
    row=3,
    column=0,
    columnspan=2,
    sticky="nswe",
    padx=5,
    pady=5)

txt_logs.insert(tk.INSERT, "Prediction")

# Mainloop the window
window.mainloop()