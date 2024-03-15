import jpegio as jio
import numpy as np
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import csv
import random

# Function to determine the quality factor
def determine_quality_factor(jpegStruct):
    qf_table = jpegStruct.quant_tables[0][0, 0]
    if qf_table == 2:
        return 95
    elif qf_table == 3:
        return 90
    elif qf_table == 8:
        return 75
    else:
        return 'Unknown'

# Define a mapping of folders to numerical classes
class_mapping = {
    'Cover': 0,
    'JMiPOD_75': 1,
    'JMiPOD_90': 2,
    'JMiPOD_95': 3,
    'JUNIWARD_75': 4,
    'JUNIWARD_90': 5,
    'JUNIWARD_95': 6,
    'UERD_75': 7,
    'UERD_90': 8,
    'UERD_95': 9
}

# Base path of the dataset
base_path = '/content/drive/MyDrive/Datasets/split1/'

# Collect all image data
image_data = []
for folder in ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']:
    folder_path = os.path.join(base_path, folder)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        jpegStruct = jpio.read(img_path)
        qf = determine_quality_factor(jpegStruct)
        img_class = class_mapping[f'{folder}_{qf}'] if folder != 'Cover' else class_mapping['Cover']
        image_data.append([img_path, img_class])

# Shuffle the data randomly
random.shuffle(image_data)

# Split the data into 75% training and 25% testing
split_index = int(0.75 * len(image_data))
training_data = image_data[:split_index]
testing_data = image_data[split_index:]

# Write training data to CSV
with open('training_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Path', 'Class'])
    writer.writerows(training_data)

# Write testing data to CSV
with open('testing_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Path', 'Class'])
    writer.writerows(testing_data)

print("Training and Testing CSV files with numerical classes have been created.")

