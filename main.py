import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.filters import sobel
from PIL import Image
from skimage.transform import rotate
from tqdm import tqdm

BASE_DIR = '/Users/mlghost/Downloads/archive/train/'
names = os.listdir(BASE_DIR)

# reading patients information and splitting into (id, image_name,label,patient_name)
label = [line.split() for line in open('/Users/mlghost/Downloads/archive/train.txt')]
df = pd.DataFrame(data=label, columns=['id', 'image_name', 'label', 'patient_name'])
os.makedirs('./dataset/', exist_ok=True)
aug_data = []
anomaly_data = []
for i in tqdm(range(len(df))):
    try:
        image_name = df.iloc[i]['image_name']
        image = plt.imread(BASE_DIR + image_name)

        S = image.shape
        if len(S) > 2:
            # converting the RGB and RGBA images to grayscale
            image = image.mean(axis=-1)

        # Central Cropping

        H, W = image.shape

        cl, cr, cb = 0.085, 0.085, 0.2

        cropped_image = image[0:-int(cb * H), int(cl * W):-int(cr * W)]

        CH, CW = 400, 400

        resized = np.array(Image.fromarray(cropped_image).resize((CH, CW)))
        plt.imsave('./dataset/' + image_name, resized, cmap='gray')
        aug_data.append([image_name, df.iloc[i]['label']])

        # if df.iloc[i]['label'] == 'positive':
        #     _90 = rotate(image, 90)
        #     _180 = rotate(image, 180)
        #     _270 = rotate(image, 270)
        #     vf = image[:, ::-1]
        #     hf = image[::-1, :]
        #     plt.imsave('./dataset/' + image_name[:image_name.index('.')] + '_90.png', _90, cmap='gray')
        #     plt.imsave('./dataset/' + image_name[:image_name.index('.')] + '_180.png', _180, cmap='gray')
        #     plt.imsave('./dataset/' + image_name[:image_name.index('.')] + '_270.png', _270, cmap='gray')
        #     plt.imsave('./dataset/' + image_name[:image_name.index('.')] + '_vf.png', vf, cmap='gray')
        #     plt.imsave('./dataset/' + image_name[:image_name.index('.')] + '_hf.png', hf, cmap='gray')
        #
        #     aug_data.append([image_name[:image_name.index('.')] + '_90.png', df.iloc[i]['label']])
        #     aug_data.append([image_name[:image_name.index('.')] + '_180.png', df.iloc[i]['label']])
        #     aug_data.append([image_name[:image_name.index('.')] + '_270.png', df.iloc[i]['label']])
        #     aug_data.append([image_name[:image_name.index('.')] + '_vf.png', df.iloc[i]['label']])
        #     aug_data.append([image_name[:image_name.index('.')] + '_hf.png', df.iloc[i]['label']])

    except Exception:
        anomaly_data.append(df.iloc[i].values)


    # plt.imshow(sobel(resized))
    # plt.show()


aug_df = pd.DataFrame(aug_data,columns=['name','label'])
aug_df.to_csv('aug.csv',index=False)

anomaly_df = pd.DataFrame(anomaly_data,columns=['id', 'image_name', 'label', 'patient_name'])
anomaly_df.to_csv('anomaly.csv',index=False)




    # plt.imshow(image)
    # plt.show()
