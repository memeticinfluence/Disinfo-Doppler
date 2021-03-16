#%% Imports
import os
import warnings

import mosaic
import pandas as pd
from tqdm import tqdm
import config
import context_manager
import data_sources.pushshift as ps
from image_utils import download_media_and_return_dhash, read_image

from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import joblib
import umap


import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import cols_conv_feats, skip_hash
from image_utils import read_image, read_and_transform_image

warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from rasterfairy import transformPointCloud2D
import time
import json
from multiprocessing import Pool
import numpy as np

def get_sql_connection():
    db = create_engine(
        "mysql://admin:coffee-admin@coffee.cp82lr4f5r06.us-east-2.rds.amazonaws.com:3306/db?charset=utf8",
        encoding="utf8",
    )
    return db

def get_subreddit_context(subreddit):
    '''
    Where will data be saved?
    '''
    sub_dir = os.path.join(config.data_dir, subreddit)
    sub_config_dir = os.path.join(config.data_dir, subreddit, "config")
    media_dir =  os.path.join(config.data_dir, 'media')
    file_subreddit = os.path.join(sub_dir, 'posts.csv.gz')
    file_subreddit_media = os.path.join(sub_dir, 'media.csv.gz')
    image_metas = os.path.join(config.data_dir, subreddit, "image_metas")
    logits_dir = os.path.join(config.data_dir, subreddit, "image_features")
    full_metadata_dir = os.path.join(config.data_dir, subreddit, "full_metadata")
    
    for _dir in [config.data_dir, sub_dir, media_dir, sub_config_dir, image_metas, logits_dir, full_metadata_dir]:
        os.makedirs(_dir, exist_ok=True)
        
    context = {
        'data_dir' : config.data_dir,
        'sub_dir' : sub_dir,
        "sub_config_dir":sub_config_dir,
        'media_dir' : media_dir,
        'file_subreddit' : file_subreddit,
        'file_subreddit_media' : file_subreddit_media,
        "image_metas": image_metas,
        "logits_dir": logits_dir,
        "full_metadata_dir": full_metadata_dir
    }
    
    return context

def download_images(row):
    preview = row.get('preview')
    context = get_subreddit_context(row['subreddit'])
    if preview:
        images = preview.get('images')
        if not images:
            return -1
        for img in images:
            r = row.copy()
            img_url, f_img = context_manager.get_media_context(img, context)
            if not img_url:
                return -1
            try:
                d_hash, img_size = download_media_and_return_dhash(img_url, f_img)
            except:
                print("ERROR!")
                return -1

            if img_size != 0:
                r['deleted'] = False
                r['d_hash'] = d_hash
                r['f_img'] = f_img 
                r['img_size'] = img_size
            else:
                r['deleted'] = True
                r['d_hash'] = d_hash
                r['f_img'] = f_img 
                r['img_size'] = img_size
            return r
    return -1


class Feature_Extraction_Dataset(Dataset):
    """Dataset wrapping images and file names
    img_col is the column for the image to be read
    index_col is a unique value to index the extracted features
    """
    def __init__(self, df, img_col, index_col):
        # filter out rows where the file is not on disk.
        self.X_train = df.drop_duplicates(subset='d_hash').reset_index(drop=True)
        self.files = self.X_train[img_col]
        self.idx = self.X_train[index_col]

    def __getitem__(self, index):
        img_idx = self.idx[index]
        img_file = self.files[index]
        try:
            img = read_and_transform_image(self.files[index], transformations)
            return img, img_file, img_idx
        except:
            pass

    def __len__(self):
        return len(self.X_train.index)

def main(input_date):
    # Are we using a GPU? If not, the device will be using cpu
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device

    

    #%% Inputs
    subreddit = "titanfolk"
    date = pd.to_datetime(input_date).date()
    next_date = date + pd.to_timedelta("1D")

    start_ux = int(time.mktime(date.timetuple()))
    end_ux = int(time.mktime(next_date.timetuple()))

    context = get_subreddit_context(subreddit)

    #%% Data & Image Collection
    new_dir = context['sub_config_dir']+"/"+input_date+".csv.gz"

    if os.path.exists(new_dir):
        df = pd.read_csv(new_dir)
    else:
        records = ps.download_subreddit_posts(subreddit, start_ux, end_ux, verbose=False)
        df = pd.DataFrame(records)
        df['preview'] = df['preview'].apply(json.dumps)
        df.to_csv(new_dir, index=False, compression='gzip')

    df['preview'] = df['preview'].fillna("{}").apply(json.loads)

    df_media = df[~df.preview.isnull()]
    df_media['subreddit'] = subreddit

    df_media = df_media.to_dict(orient='rocords')

    image_meta_dir = context['image_metas']+"/"+input_date+".csv.gz"

    if os.path.exists(image_meta_dir):
        _df_img_meta = pd.read_csv(image_meta_dir)
    else:
        bar = tqdm(total=len(df_media), desc="Downloading images")
        img_meta = []
        p = Pool()
        for o in p.imap_unordered(download_images, df_media):
            if o != -1:
                img_meta.append(o)
            bar.update(1)
        _df_img_meta = pd.DataFrame(img_meta).drop('subreddit', axis=1)
        _df_img_meta.to_csv(image_meta_dir, index=False, compression='gzip')
    
    _df_img_meta = _df_img_meta[~_df_img_meta['d_hash'].isin(skip_hash)] 

    # The image needs to be specific dimensions, normalized, and converted to a Tensor to be read into a PyTorch model.
    scaler = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # this is the order of operations that will occur on each image.
    transformations = transforms.Compose([scaler, 
                                        to_tensor, 
                                        normalizer])

    class Feature_Extraction_Dataset(Dataset):
        """Dataset wrapping images and file names
        img_col is the column for the image to be read
        index_col is a unique value to index the extracted features
        """
        def __init__(self, df, img_col, index_col):
            # filter out rows where the file is not on disk.
            self.X_train = df.drop_duplicates(subset='d_hash').reset_index(drop=True)
            self.files = self.X_train[img_col]
            self.idx = self.X_train[index_col]

        def __getitem__(self, index):
            img_idx = self.idx[index]
            img_file = self.files[index]
            try:
                img = read_and_transform_image(self.files[index], transformations)
                return img, img_file, img_idx
            except:
                pass

        def __len__(self):
            return len(self.X_train.index)

    dataset = Feature_Extraction_Dataset(_df_img_meta, 
                                        img_col='f_img', 
                                        index_col='d_hash')
    data_loader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=0)

    def load_resnet_for_feature_extraction():
        # Load a pre-trained model
        res50_model = models.resnet50(pretrained=True)
        # Pop the last Dense layer off. This will give us convolutional features.
        res50_conv = nn.Sequential(*list(res50_model.children())[:-1])
        res50_conv.to(device)

        # Don't run backprop!
        for param in res50_conv.parameters():
            param.requires_grad = False

        # we won't be training the model. Instead, we just want predictions so we switch to "eval" mode. 
        res50_conv.eval();
        
        return res50_conv

    res50_conv = load_resnet_for_feature_extraction()

    logits_dir = context['logits_dir']+"/"+input_date+".csv.gz"

    conv = None

    if os.path.exists(logits_dir):
        conv = pd.read_csv(logits_dir, index_col=0).drop_duplicates()

    df_convs = []
    for (X, img_file, idx) in tqdm(data_loader):
        filt = [i for i in idx if i not in conv.index] if conv is not None else [i for i in idx]
        if filt:
            X = X.to(device)
            logits = res50_conv(X)
            #logits.size() # [`batch_size`, 2048, 1, 1])

            logits = logits.squeeze(2) # remove the extra dims
            logits = logits.squeeze(2) # remove the extra dims
            #logits.size() # [`batch_size`, 2048]

            n_dimensions = logits.size(1)
            logits_dict = dict(zip(idx, logits.cpu().data.numpy()))
            #{'filename' : np.array([x0, x1, ... x2047])}

            df_conv = pd.DataFrame.from_dict(logits_dict, 
                                            columns=cols_conv_feats, 
                                            orient='index')
            # add a column for the filename of images...
            df_conv['f_img'] = img_file
            df_conv = df_conv.loc[filt]
            df_convs.append(df_conv)    

    conv = pd.concat([conv, *df_convs]).drop_duplicates()

    conv.to_csv(logits_dir, compression='gzip')

    # UMAP Params
    n_neighbors = 25
    metric = 'euclidean'
    min_dist = 0.5
    training_set_size = config.umap_training_set_size
    overwrite_model = False # set to True to re-train the model.
    os.makedirs(f'{ config.working_dir }/encoders',exist_ok=True)
    os.makedirs(f'{ config.working_dir }/umap_training_data',exist_ok=True)
    # Model files
    file_encoder = (f'{ config.working_dir }/encoders/{ str(min_dist).replace(".", "-") }_'
                    f'dist_{ metric }_sample_{ training_set_size }_{input_date}.pkl')
    file_training_set = f'{ config.working_dir }/{ training_set_size }_{input_date}.csv'

    if not os.path.exists(file_encoder) or overwrite_model:
        # Create the training set (note: UMAP can be either supervised or unsupervised.)
        if not os.path.exists(file_training_set):
            training_set = conv[config.cols_conv_feats].sample(training_set_size, 
                                                                random_state=303)
        else:
            training_set = pd.read_csv(file_training_set, 
                                    index_col=0)
        
        # fit the model scikit-learn style
        encoder = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            random_state=303,
                            verbose=1).fit(training_set.values)

        # save the model for later! Save the training data, too.
        joblib.dump(encoder, file_encoder)                             
        training_set.to_csv(file_training_set)
    else:
        encoder = joblib.load(file_encoder)
        encoder

    logits_dir = context['full_metadata_dir']+"/"+input_date+".csv.gz"

    _df_img_meta

    # Join the image metadata with convolutional features
    if not os.path.exists(logits_dir):
        # Merge the datasets
        merge_cols = [c for c in _df_img_meta.columns if c != 'f_img']
        df_merged = pd.merge(left=_df_img_meta[merge_cols],
                            right=conv.reset_index(), 
                            how='left',
                            left_on='d_hash',
                            right_on='index')
        df_merged.to_csv(logits_dir, 
                        compression='gzip')
    else:
        df_merged = pd.read_csv(logits_dir, 
                                index_col=0, 
                                compression='gzip')

    tile_width, tile_height = config.tile_width, config.tile_height # pixel dimenstions per image

    nx = config.mosiac_width # number of images in the x and y axis
    ny = df_merged.shape[0] // nx
    sample_size = nx * ny
    aspect_ratio = float(tile_width) / tile_height

    # sample the dataset
    df_sample = df_merged.sample(sample_size, random_state=303)
    images = df_sample.f_img
    embeddings = encoder.transform(df_sample[config.cols_conv_feats].replace([-np.inf, np.inf], np.nan).fillna(0).values)



    grid_assignment = transformPointCloud2D(embeddings, 
                                                target=(nx, 
                                                        ny))

    

    def get_sql_connection():
        db = create_engine(
            "mysql://admin:65Qyw6pgSo8F3LyiASFr@meme-observatory.cizj1wczwqh5.us-west-2.rds.amazonaws.com:3306/meme_observatory?charset=utf8",
            encoding="utf8",
        )
        return db

    db = get_sql_connection()
    wanted_cols = ["url", "full_link", "x", "y"]
    df_sample[['x', 'y']] = grid_assignment[0].astype(int)

    output_df = df_sample[wanted_cols]
    output_df['subreddit'] = subreddit
    output_df['dt'] = pd.to_datetime(input_date)
    output_df.to_sql("mosaics", db, index=False, if_exists="append")
    

if __name__=="__main__":
    dates = ['20201231', '20210101', '20210102', '20210103', '20210104', '20210105',
       '20210106', '20210107', '20210108', '20210109', '20210110', '20210111',
       '20210112', '20210113', '20210114', '20210115', '20210116', '20210117',
       '20210118', '20210119', '20210120', '20210121', '20210122', '20210123',
       '20210124', '20210125', '20210126', '20210127', '20210128', '20210129',
       '20210130']
    for d in dates:
        main(d)