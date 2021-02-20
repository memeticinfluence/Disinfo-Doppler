import os
import datetime

# change this!
subreddit = 'dankmemes'
data_dir = 'data/platforms/reddit'
working_dir = os.path.join(data_dir, f'{subreddit}/')
media_dir = os.path.join(data_dir, 'media')
mosaic_dir = os.path.join(working_dir, 'mosaics')
output_dir = 'ouput/'
image_lookup_file = os.path.join(working_dir, 'media.csv.gz')

# these files don't exist yet
logits_file = os.path.join(working_dir, 'image_features.csv.gz')
full_metadata_file = os.path.join(working_dir, 'full_metadata.csv.gz')
sample_dataset_file = os.path.join(working_dir, 'sample_media.csv.gz')
two_dim_embeddings_file = os.path.join(working_dir, '2d_embeddings.csv')
file_animation = os.path.join(output_dir,'doppler_mosaic.mp4')

for _dir in [working_dir, media_dir, output_dir, mosaic_dir]:
    os.makedirs(_dir, exist_ok=True)
    
# shared variables
skip_hash = ['NOHASH', '0000000000000000', 'nan']
n_dimensions = 2048 # features from resnet50, change this is you change the model in feature extraction.
cols_conv_feats = [f'conv_{n}' for n in range(n_dimensions)]
