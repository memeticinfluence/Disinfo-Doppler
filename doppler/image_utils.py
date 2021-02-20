import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import shutil

from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from torchvision import transforms


s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
s.mount('http://', HTTPAdapter(max_retries=retries))


def resize_image(tile, tile_width, tile_height, aspect_ratio=False):
    '''
    Crops and centers images for the mosaic.
    '''
    tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
    if (tile_ar > aspect_ratio):
        margin = 0.5 * (tile.width - aspect_ratio * tile.height)
        tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, 
                          tile.height))
    else:
        margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
        tile = tile.crop((0, margin, tile.width, 
                          margin + float(tile.width) / aspect_ratio))
    tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
    
    return tile

def read_image(img_file):
    '''
    Reads an image either from memeory or from the web!
    '''
    if os.path.exists(img_file):
        # from disk...
        img = Image.open(img_file)
    else:
        # from the web...
        r = requests.get(img_file, stream=True)
        img = Image.open(BytesIO(r.content))
    
    return img

def read_and_transform_image(img_file, transformations=None):
    '''This function reads an image into a Pillow object,
    converts it to a PNG, and transforms it into a Torch Tensor'''
    img = read_image(img_file)

    # convert to png
    img = img.convert('RGB')
    
    # transform to tensor
    if isinstance(transformations, transforms.Compose):
        img = transformations(img)
    
    return img


def display_n_images(imgs, per_row=4):
    '''
    Plots a row of images
    '''
    for i in range(len(imgs)):
        if i % per_row == 0:
            _ , ax = plt.subplots(1, per_row, sharex='col', sharey='row', figsize=(24, 6))
        j = i % per_row
        image =  imgs[i]
        image = resize_img(image, tile_height=244, tile_width=244)
        ax[j].imshow(image)
        ax[j].axis('off')
        
def download_media(url:str, 
                   fn:str):
    '''
    Downloads an image from the open web.
    '''
    # Download the image
    r = s.get(url, stream=True)
    if not r.status_code == 200:
        return False
    
    # download the image locally
    with open(fn, 'wb') as file:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, file)
    
    return True

def download_media_and_return_dhash(url, fn):
    '''
    Downloads a media file from the open web to local disk (if it does not exist yet).
    Calculates a dhash and the size of the file, and returns both as a tuple.
    If the download fails, returns a placeholder 'NOHASH' and a filezie of 0.
    '''
    if os.path.exists(fn):
        # is th image exists, don't download it again.
        # calculate the size and hash
        img_size = os.path.getsize(fn)
        if img_size != 0:
            # read the image and calculate the hash
            img = read_image(fn)
            dhash = str(imagehash.dhash(img, hash_size=8))
            
            return dhash, img_size
    else:
        if download_media(url, fn):
            # calculate the hash
            img = read_image(fn)
            img_size = os.path.getsize(fn)
            dhash = str(imagehash.dhash(img, hash_size=8))

            return dhash, img_size
    
    return 'NOHASH', 0

