# geoguessr-ai
ML project revolving around guessing position by looking at street view image. 

### Approach
- Download as many street view images as possible, trim and compress
- Create classes using S2 library ![](classes_7340.png) Used similar parition technique as this paper: https://research.google.com/pubs/archive/45488.pdf ![image](https://github.com/hub2/geoguessr-ai/assets/5579578/b8dcadca-1c91-4cd3-8f0c-173590eb50c2)

- Split dataset into classes, try to get balanced datasets
- Download a lot more of street view images for small classes, islands, cities and alike (not everything is on google street view as you can see): ![](panoramas.png)![image](https://github.com/hub2/geoguessr-ai/assets/5579578/f64e89a7-e7ca-4c2b-9627-884c2c94fd63)

- Example image ![](download_panoramas/example.png)
- Train ResNet-50 with pretrained weights (changed first and last layer to accomodate input and output) using automated script that deploy code and train on https://runpod.io
- Run, test, repeat training. Add well-known augmentations, custom augmentation (PanoramaShifting), more trimming, lower resolution etc.
- Used GradCamPlusPlus (check_net.py) to see what ResNet is looking at to make it's decisions ![image](https://github.com/hub2/geoguessr-ai/assets/5579578/5ef85323-09ac-4fc8-8704-7b698844ea63) ![image](https://github.com/hub2/geoguessr-ai/assets/5579578/dd188612-9b55-4c9a-966a-e04de21153da)


- My best attempt (~geoguessr Master level) used "voting algorithm" on images from multiple years of street view

### Things I want to try
- Add multiple heads to guess country/continent and location
