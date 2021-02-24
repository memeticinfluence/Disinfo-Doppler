import os
import config

def get_context(subreddit):
    '''
    Where will data be saved?
    '''
    sub_dir = os.path.join(config.data_dir, subreddit)
    media_dir =  os.path.join(config.data_dir, 'media')
    file_subreddit = os.path.join(sub_dir, 'posts.csv.gz')
    file_subreddit_media = os.path.join(sub_dir, 'media.csv.gz')
    
    for _dir in [config.data_dir, sub_dir, media_dir]:
        os.makedirs(_dir, exist_ok=True)
        
    context = {
        'data_dir' : config.data_dir,
        'sub_dir' : sub_dir,
        'media_dir' : media_dir,
        'file_subreddit' : file_subreddit,
        'file_subreddit_media' : file_subreddit_media
    }
    
    return context


def get_media_context(image, context):
    '''
    Establishes where media files will be saved.
    '''
    image_id = image['id']
    pos_images = image.get('resolutions')
    if pos_images:
        largest_image = pos_images[-1]
    else:
        # no images...
        return None, None
    
    # where is the image to be downlaoded?
    img_url = largest_image.get('url')
    img_url = img_url.replace('&amp;', '&')
    
    # what is the file  extension?
    _, ext = os.path.splitext(img_url.split('?')[0])
    ext = ext.replace('jpeg', 'jpg')
    
    # where will the images be downloaded locally?
    dir_img = os.path.join(context['media_dir'], 
                           image_id[:2].lower(), 
                           image_id[2:4].lower())
    
    f_img = os.path.join(dir_img, image_id + ext)
    os.makedirs(dir_img, exist_ok=True)
    
    return img_url, f_img