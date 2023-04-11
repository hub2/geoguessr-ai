import streetview
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, shape
import random
import sys
import fiona


black_list = ["FR"]
def generate_coordinate(land_areas):
    while True:
        lon_start, lat_start, lon_end, lat_end  =  [54.2245,24.1393,56.1691,26.1185]
        lat = random.uniform(lat_start, lat_end)
        lon = random.uniform(lon_start, lon_end)

        point = Point(lon, lat)
        return ((lat, lon))

        # Check if point is inside any of the land areas
        is_on_land = False
        for iso, land_area in land_areas:
            if iso not in black_list and land_area.contains(point):
                is_on_land = True
                break

        if is_on_land:
            return((lat, lon))


land_areas = []
with fiona.open("../TM_WORLD_BORDERS/TM_WORLD_BORDERS-0.3.shp", "r") as shapefile:
    for feature in shapefile:
        land_areas.append((feature["properties"]["ISO2"], shape(feature["geometry"])))


while True:
    lat, lon = generate_coordinate(land_areas)
    #print(".", end="")
    sys.stdout.flush()
    panoids = streetview.panoids(lat=lat, lon=lon)
    if len(panoids) == 0:
        continue

    pano = random.choice(panoids)
    lat = pano['lat']
    lon = pano['lon']
    panoid = pano['panoid']
    name = str(lat) + "_" + str(lon) + ".jpg"

    panorama = streetview.download_panorama_v3(panoid, zoom=2, disp=False)
    print("\nsaving... " + name)
    panorama.save(os.path.join("downloads", name))
