import os
import pandas as pd
import numpy as np
import cv2

def csv_to_images(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    for idx, (label, pixel_values) in enumerate(zip(labels, pixels)):
        img = np.reshape(pixel_values, (28, 28)).astype("uint8")
        filepath = os.path.join(output_dir, str(label), f"{idx}.png")
        cv2.imwrite(filepath, img)

    print(f"Saved all images to: {output_dir}")

# ---- RUN THIS ----
csv_to_images(r"C:\Users\chait\OneDrive - BENNETT UNIVERSITY\Documents\fMRI\autoencoders\mnist_train.csv", "data/train/images")
csv_to_images(r"C:\Users\chait\OneDrive - BENNETT UNIVERSITY\Documents\fMRI\autoencoders\mnist_test.csv", "data/test/images")
