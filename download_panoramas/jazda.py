import streetview
import matplotlib.pyplot as plt

panoids = streetview.panoids(lat=40.75388056, lon=-73.99697222)
panoid = panoids[0]['panoid']
panorama = streetview.download_panorama_v3(panoid, zoom=2, disp=False)
plt.imshow(panorama)
panorama.save("dupa.png")
