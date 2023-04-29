import streetview
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, shape
import random
import sys
import fiona
import csv


ccs = [
    #[54.2245,24.1393,56.1691,26.1185], # ?
    #[-70.811921,-33.62845,-70.509797,-33.344404], # santiago
    #[105.532931,-10.570013,105.726222,-10.40966], # christmas islands
    #[-67.256946,17.908335,-65.61449,18.544873], # puerto rico
    [-168.2,-56.3,179.5,76.4], # world
    #[166.57,-47.33,178.81,-33.94], # NZ
    #[125.9329,34.1788,129.6683,38.6158], # south korea
    [-18.6,-35.1,51.0,36.2], # Africa
    [-18.6,-35.1,51.0,36.2], # Africa
    [-18.6,-35.1,51.0,36.2], # Africa
    #[-80.7,-54.3,-35.0,12.5], # South America
    #[94.6,-44.4,154.8,20.9], #Asia + Australia
]

with open("worldcities.csv") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    header = next(reader)
    for line in reader:
        lat, lon = float(line[2]), float(line[3])
        pop = line[9]
        if pop:
            pop = float(pop)
        else:
            pop = 0

        if pop < 50000:
            continue

        bbox_off = min(max(pop/37732000 * 1, 0.02), 0.3)

        coords = [lon-bbox_off, lat-bbox_off, lon+bbox_off, lat+bbox_off]
        #print(coords)
        #ccs.append(coords)

black_list = []
def generate_coordinate(land_areas):
    while True:
        lon_start, lat_start, lon_end, lat_end  = random.choice(ccs)
        if lat_end < lat_start:
            lat_start, lat_end = lat_end, lat_start
        if lon_end < lon_start:
            lon_start, lon_end = lon_end, lon_start

        lat = random.uniform(lat_start, lat_end)
        lon = random.uniform(lon_start, lon_end)

        point = Point(lon, lat)

        # Check if point is inside any of the land areas
        is_on_land = False
        for iso, land_area in land_areas:
            if iso not in black_list and land_area.contains(point):
                is_on_land = True
                break

        if is_on_land:
            try:
                panoids = streetview.panoids(lat=lat, lon=lon)
                #panoids = [random.choice(panoids)]
                random.shuffle(panoids)
                panoids = panoids[:2]
            except Exception:
                continue
            if len(panoids) != 0:
                return panoids


land_areas = []
with fiona.open("../TM_WORLD_BORDERS/TM_WORLD_BORDERS-0.3.shp", "r") as shapefile:
    for feature in shapefile:
        land_areas.append((feature["properties"]["ISO2"], shape(feature["geometry"])))


coordinate_queue = Queue(maxsize=10)

def coordinate_generator():
    while True:
        panoids = generate_coordinate(land_areas)
        coordinate_queue.put(panoids)


def downloader():
    while True:
        panoids = coordinate_queue.get()

        sys.stdout.flush()
        if len(panoids) == 0:
            continue
        print(f"Saving {len(panoids)} images, queue {coordinate_queue.qsize()}")

        for idx, pano in enumerate(panoids):
            lat = pano['lat']
            lon = pano['lon']
            panoid = pano['panoid']
            name = str(lat) + "_" + str(lon) + "_" + str(idx) + ".jpg"
            try:
                panorama = streetview.download_panorama_v3(panoid, zoom=2, disp=False)
            except Exception:
                continue
            #print("\nsaving... " + name)
            panorama.save(os.path.join("/media/des/Data2tb/geoguessr/", name))


downloaders = []
generators = []

for down in range(10):
    downloader_thread = Process(target=downloader)
    downloader_thread.start()
    downloaders.append(downloader_thread)

for gen in range(8):
    generator_thread = Process(target=coordinate_generator)
    generator_thread.start()
    generators.append(generator_thread)


for down in downloaders:
    down.join()

for generator_thread in generators:
    generator_thread.join()

