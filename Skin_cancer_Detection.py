from fastai import *
from fastai.vision import *
from fastai.callbacks import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("data/HAM10000_metadata.csv")
df.head()

# Categories of the diferent diseases
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

df['lesion'] = df.dx.map(lesion_type_dict)
df.head()

print(df.lesion.value_counts())

fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
sns.countplot(y='lesion',data=df, hue="lesion",ax=ax1)

num_sample = 200

df_df = df.loc[df['dx'] == "df"][0:115]
df_vasc = df.loc[df['dx'] == "vasc"][0:142]
df_akiec = (df.loc[df['dx'] == "akiec"]).sample(num_sample)
df_bcc = df.loc[df['dx'] == "bcc"][0:num_sample].sample(num_sample)
df_bkl = df.loc[df['dx'] == "bkl"][0:num_sample].sample(num_sample)
df_mel = df.loc[df['dx'] == "mel"][0:num_sample].sample(num_sample)
df_nv = df.loc[df['dx'] == "nv"][0:num_sample].sample(num_sample)

df = pd.concat([df_akiec, df_bcc, df_bkl, df_df, df_mel, df_nv, df_vasc])
df = shuffle(df)

fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
sns.countplot(y='lesion',data=df, hue="lesion",ax=ax1)
plt.show()

# tfms = get_transforms(flip_vert=True)
# data = ImageDataBunch.from_df("/Python Projects/SkinCancerDetection-WebApp-master/data/HAM10000_images_part_2/", df, fn_col=1, suffix='.jpg', label_col=7, ds_tfms=tfms, size=224, bs=16)
# data.normalize(imagenet_stats)
#
# data.show_batch(rows=3)
#
# learner = cnn_learner(data, models.densenet169, metrics=[accuracy, FBeta(average='macro')], model_dir="../models/")
# learner.loss_func = nn.CrossEntropyLoss()
#
# learner.lr_find()
# learner.recorder.plot()
#
# learner.fit_one_cycle(30, 1e-3, callbacks=[SaveModelCallback(learner, every='improvement', monitor='accuracy', name='model_best')])
#
# learner = learner.load("model_best")
#
# interp = ClassificationInterpretation.from_learner(learner)
# interp.plot_confusion_matrix(figsize=(10,8))
#
#
# interp.most_confused()
# interp.plot_top_losses(9, figsize=(15,10), heatmap=False)

