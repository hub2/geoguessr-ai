import streetview
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, shape
import random
import sys
import fiona

def generate_coordinate(land_areas):
    while True:
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)

        point = Point(lon, lat)

        # Check if point is inside any of the land areas
        is_on_land = False
        for land_area in land_areas:
            if land_area.contains(point):
                is_on_land = True
                break

        if is_on_land:
            return((lat, lon))


land_areas = []
with fiona.open("../TM_WORLD_BORDERS/TM_WORLD_BORDERS-0.3.shp", "r") as shapefile:
    for feature in shapefile:
        land_areas.append(shape(feature["geometry"]))


while True:
    lat, lon = generate_coordinate(land_areas)
    name = str(lat) + "_" + str(lon) + ".jpg"
    print(".", end="")
    sys.stdout.flush()
    panoids = streetview.panoids(lat=lat, lon=lon)
    if len(panoids) == 0:
        continue

    panoid = panoids[0]['panoid']
    panorama = streetview.download_panorama_v3(panoid, zoom=2, disp=False)
    print("\nsaving... " + name)
    panorama.save(os.path.join("downloads", name))
