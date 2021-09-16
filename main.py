import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.filters import sobel

BASE_DIR = '/Users/mlghost/Downloads/archive/train/'
names = os.listdir(BASE_DIR)

# reading patients information and splitting into (id, image_name,label,patient_name)
label = [line.split() for line in open('/Users/mlghost/Downloads/archive/train.txt')]
df = pd.DataFrame(data=label, columns=['id', 'image_name', 'label', 'patient_name'])

pixel_number = []
for i in range(len(df)):
    try:
        image_name = df.iloc[i]['image_name']
        image = plt.imread(BASE_DIR + image_name)
        pixel_number.append(image.shape[0] * image.shape[1])
    except Exception:
        print(image_name)
plt.hist(pixel_number,bins=4)
plt.show()


# # for i in range(len(df)):
# #     image_name = df.iloc[i]['image_name']
# #     image = plt.imread(BASE_DIR + image_name)
# #     print(image.shape)
#
# pos_data = df[df['label'] == 'positive']
# neg_data = df[df['label'] == 'negative']
#
# fig, ax = plt.subplots(2, 5)
# os.makedirs('samples/',exist_ok=True)
# for i in range(5):
#     image_name = pos_data.iloc[i]['image_name']
#     image = plt.imread(BASE_DIR + image_name)
#     # plt.imsave('./samples/pos_' + str(i) + '.png', image)
#     ax[0][i].imshow(image)
#     ax[0][i].axis('off')
#
# for i in range(5):
#     image_name = neg_data.iloc[40 + i]['image_name']
#     image = plt.imread(BASE_DIR + image_name)
#     # plt.imsave('./samples/neg_' + str(i) + '.png', image)
#     ax[1][i].imshow(image)
#     ax[1][i].axis('off')
#     # plt.imsave(str(i)+'.png', sobel(image))
# plt.show()
"""
90,180,270 + original + flipping = 2158*6

13000 + 13000 = 26000
"""
