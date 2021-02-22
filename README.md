# Disinfo Doppler
### Create Mosaics of Imagery in Online Communities Over Time

![](https://github.com/memeticinfluence/Disinfo-Doppler/blob/master/samples/header.gif?raw=true)


**Initial Code by Leon Yin**<br>
Published: 2019-04-06<br>
<br>
**Dockerized and maintained by:**<br>
Jansen Derr - [memetic influence](www.memeticinfluence.com)<br>
Last Updated: 2021-02-21

## Installation Instructions:

1. Download [Docker for Desktop](https://www.docker.com/products/docker-desktop)
2. Clone this repo to your local machine and CD into Repo in terminal or CMD
3. Run `docker-compose up` 
4. When successfully started, a `URL` should appear in your terminal window. Copy and Paste this into a browser.

To update, simply re-download the repo and run `docker-compose build` to rebuild.

Configure the subreddit to download from in `config.py`.

## Intro
The Doppler is an open source computer vision toolkit used to trace and measure image-based activity online. These notebooks download imagery from various sources, such as subreddits, and then creates image-clustered mosaics over time. This helps us understand, visually, what sorts of source content and edits are being spread into an online community. 

## How does it work?
In order to use the two functions of the Doppler, images need to be transformed into differential features. To do this we use two computer vision techniques, d-hashing and feature extraction using a pre-trained neural network.

### D-Hashing
D-Hashing creates a fingerprint for an image (regardless of size or minor color variations). With this technique it is easy to check for duplicate images. This method (outlined [here](http://www.hackerfactor.com/blog/?/archives/529-Kind-of-Like-That.html)) is also quick and not intense for a computer. We use the imagehash Python library to do this.

### Feature Extraction
Neural networks are able to learn numeric attributes used to differentiate between images. These numbers are continuous which allows us to calculate similarity. These features are what allow us to cluster images for the mosaic and rank similarity for the reverse image search. Thanks for the decidcation of open source developers and researchers implementation is relatively easy. However, it requires a lot of matrix math which is a lot of work for a regular computer. This process is greatly accelerated using a computer with a graphics processing unit (GPU). We use PyTorch to do this.

These two techniques serve somewhat different purposes. The Doppler architecture intends to take advantage of both techniques when appropriate.

## Why We're Building this?
We seek to empower newsrooms, researchers and members of civil society groups to easily research and debunk coordinated hoaxes, harassment campaigns, and racist propaganda that originate from unindexed online commuities.

Specifically, the Disinfo Doppler will help evidence-based reporting and research into content that is ephemeral, unindexed and toxic in nature. The Disinfo Doppler would allow a greater variety of users the ability to navigate and investigate these spaces in a more secure and systematic way than is currently available. Formalizing how we observse this content is of utmost importance, as extended contact with these spaces is unnecessary and can lead to vicarious trauma, and in some rare cases radicalization. The Disinfo Doppler allows users to distance themselves from tertiary material not relevant to their investigation, while providing context vertically and horizontally.

![](https://github.com/memeticinfluence/logos/blob/main/horizontal_transparent.png?raw=true)
