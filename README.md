<p align="center">
  <img width="460" height="300" src="https://github.com/memeticinfluence/meme-monitor/blob/master/samples/header.gif?raw=true">
</p>

![](https://github.com/memeticinfluence/meme-monitor/blob/master/samples/description.png?raw=true)

# Create Mosaics of Imagery in Online Communities Over Time<br>

## Installation Instructions:

1. **Download** [Docker for Desktop](https://www.docker.com/products/docker-desktop) - restart your computer after installation and **make sure it is running** before continuing. In Docker Desktop, especially on Macs, please increase the amount of memory available to at least 8GB.
2. **Clone** this repo to your local machine and CD into Repo in terminal or CMD
3. **Run** `docker-compose build` to build (required first boot/update) and `docker-compose up` to start server 
4. **Copy and Paste** the Jupyter server notebook `URL` from Terminal into your local web browser.

To **update**, simply `git pull` the repo and run `docker-compose build` to rebuild. You should end with `Successfully tagged meme-monitor_jupyter:latest` when successfully built or rebuilt.

**Configure** settings such as the subreddit to download from in `config.py`.

### Not working?
**Not Starting:** Make sure `Docker Desktop` is running - pull the repo and run `docker-compose build`<br> -  Try `Kernel > Restart and Clear Output` in Jupyter<br>
**Download Troubles:** clear `/data/platforms/reddit/` of all files - make sure folder structure `/data/platforms/reddit/<subreddit-name>/` exists<br>
**Rendering Issues:**  clear the `mosaics` folder in `/data/platforms/reddit/<subreddit-name>` - create the `mosaics` folder in `/<subreddit-name>` if it doesn't exist - clear all `.gz` files in `/<subreddit-name>` and restart the whole process<br>

### Need Help?
**Email us:** code@memeticinfluence.com<br>
**Tweet us:** [@memeinfluence](https://twitter.com/memeinfluence)<br>
**Text us:**  855-420-MEME (6363)<br>


## **Intro from [Leon Yin](https://github.com/yinleon/Disinfo-Doppler)**<br>
The Meme Monitor is an open source computer vision toolkit used to trace and measure image-based activity online. These notebooks download imagery from various sources, such as subreddits, and then creates image-clustered mosaics over time. This helps us understand, visually, what sorts of source content and edits are being spread into an online community. 

## How does it work?
In order to use the two functions of the Meme Monitor, images need to be transformed into differential features. To do this we use two computer vision techniques, d-hashing and feature extraction using a pre-trained neural network.

### D-Hashing
D-Hashing creates a fingerprint for an image (regardless of size or minor color variations). With this technique it is easy to check for duplicate images. This method (outlined [here](http://www.hackerfactor.com/blog/?/archives/529-Kind-of-Like-That.html)) is also quick and not intense for a computer. We use the imagehash Python library to do this.

### Feature Extraction
Neural networks are able to learn numeric attributes used to differentiate between images. These numbers are continuous which allows us to calculate similarity. These features are what allow us to cluster images for the mosaic and rank similarity for the reverse image search. Thanks for the decidcation of open source developers and researchers implementation is relatively easy. However, it requires a lot of matrix math which is a lot of work for a regular computer. This process is greatly accelerated using a computer with a graphics processing unit (GPU). We use PyTorch to do this.

These two techniques serve somewhat different purposes. The Meme Monitor architecture intends to take advantage of both techniques when appropriate.

## Why We're Building this?
We seek to empower newsrooms, researchers and members of civil society groups to easily research and debunk coordinated hoaxes, harassment campaigns, and racist propaganda that originate from unindexed online commuities.

Specifically, the Meme Monitor will help evidence-based reporting and research into content that is ephemeral, unindexed and toxic in nature. The Meme Monitor would allow a greater variety of users the ability to navigate and investigate these spaces in a more secure and systematic way than is currently available. Formalizing how we observse this content is of utmost importance, as extended contact with these spaces is unnecessary and can lead to vicarious trauma, and in some rare cases radicalization. The Meme Monitor allows users to distance themselves from tertiary material not relevant to their investigation, while providing context vertically and horizontally.

## Credits - visit us at www.memeticinfluence.com

**Developed by:**<br>
[Jansen Derr](https://github.com/jansenderr) - [memetic influence](https://www.memeticinfluence.com)<br>
Last Updated: 2021-02-26
<br>
![](https://github.com/memeticinfluence/logos/blob/main/horizontal_transparent_small.png?raw=true)
<br>
**Technique by:**<br>
[Leon Yin](https://github.com/yinleon/Disinfo-Doppler) + [Dr. Joan Donovan](https://www.hks.harvard.edu/faculty/joan-donovan) - [HKS Shorenstein Center - TaSC](https://shorensteincenter.org/programs/technology-social-change/)<br>
Published: 2019-04-06<br><br>
![](https://github.com/memeticinfluence/meme-monitor/blob/master/samples/credits/Harvard%20Kennedy%20School%20-%20Shorenstein%20Center/logo_small.png?raw=true)
<br>
<br>


<br>


