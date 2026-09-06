import numpy as np, csv
from PIL import Image

labels = np.loadtxt("data/labels.csv", dtype=int)
# grab the first sample of a few different classes
wanted = [12, 17, 45, 62]  
saved = {}
with open("data/images.csv") as f:
    for i, row in enumerate(csv.reader(f)):
        lbl = labels[i]
        if lbl in wanted and lbl not in saved:
            img = np.array(row, dtype=np.uint8).reshape(32, 32)
            Image.fromarray(img).resize((200,200)).save(f"test_class_{lbl}.png")
            saved[lbl] = True
        if len(saved) == len(wanted):
            break
print("saved:", list(saved.keys()))