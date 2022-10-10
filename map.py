import os
import json
import glob
import numpy as np
import shapely.geometry as sg
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

#DATASET_PATH = "C:\\Users\\huber\\Programowanie\\geoguessr\\data_panoramas"
DATASET_PATH = "/media/des/742C46F62C46B342/Users/huber/Programowanie/geoguessr/data_panoramas"

cars = glob.glob(os.path.join(DATASET_PATH, "*_car.png"))
pans = glob.glob(os.path.join(DATASET_PATH, "*_pan.png"))


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
pngs = list(zip(pans, cars))
# print(pngs)

dataset = []
for idx, item in enumerate(pngs):
    json_basename = os.path.basename(item[0])
    json_filename = json_basename.split("_")[0]
    with open(os.path.join(DATASET_PATH, json_filename + ".json"), "r") as f:
        info = json.loads(f.read().replace("'", "\""))
        lat = info["lat"]
        lng = info["lng"]
        dataset.append((lng, lat))


def show_coords_on_map(dataset):
    df = pd.DataFrame(dataset, columns=["Longitude", "Latitude"])
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)

    #this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    print(world)
    for point in geometry:
        for i, row in world.iterrows():
            country = row["name"]
            row = world.iloc[[i]]
            shape = GeoDataFrame(row)
            shape = shape["geometry"].unary_union
            if point.intersects(shape):
                print(country)
    gdf.plot(ax=world.plot(figsize=(20, 12)), markersize=1)
    plt.show()

def get_middle_point(poly):
    return ((poly[0][0]+ poly[2][0])/2, (poly[0][1]+poly[2][1])/2)

# show_coords_on_map(dataset)

lon_space = np.linspace(-180, 180, 60).tolist()
lat_space = np.linspace(-90, 90, 30).tolist()

print(lon_space)
print(lat_space)
polygons = []

lon_len = len(lon_space)
lat_len = len(lat_space)
for idx, lon in list(enumerate(lon_space))[:-1]:
    for idx2, lat in list(enumerate(lat_space))[:-1]:
        next_lon = lon_space[(idx+1)]
        next_lat = lat_space[(idx2+1)]
        polygons.append(((lon, lat), (next_lon, lat), (next_lon, next_lat), (lon, next_lat)))


out = []
for pol in polygons:
    polygon = sg.Polygon(pol)
    print(polygon)
    print(pol)
    #geoframe = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
    out.append(polygon)

gdf = gpd.GeoSeries(out)
#gdf.plot(ax=world.plot(figsize=(20, 12)), marker='o', color='red', markersize=1)
#plt.show()

df = pd.DataFrame(dataset, columns=["Longitude", "Latitude"])
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

polys_with_stuff = []
for geom in geometry:
    for idx, poly in enumerate(gdf):
        if geom.intersects(poly):
            if poly not in polys_with_stuff:
                polys_with_stuff.append(poly)
            break

gdf = gpd.GeoSeries(polys_with_stuff)
print(gdf)

classes = list(enumerate(gdf.values.tolist()))

