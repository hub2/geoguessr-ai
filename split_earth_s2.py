import pywraps2 as pys2
import s2cell
import os, re
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString

# Set the path to the "downloads" folder
path = "./download_panoramas/downloads"


# Create an empty list to store the coordinates
coords = []

# Loop over the files in the "downloads" folder
for filename in os.listdir(path):
    # Check if the file is a JPG image
    if filename.endswith(".jpg"):
        # Extract the latitude and longitude from the filename
        out = filename.split(".jpg")[0]
        lon, lat = out.split("_")
        lat = float(lat)
        lon = float(lon)
        coords.append((lat, lon))

max_lat = 0
max_lon = 0
for coord in coords:
    if coord[0] > max_lat:
        max_lat = coord[0]
    if coord[1] > max_lon:
        max_lon = coord[1]

locations_all = {}
for lat, lon in coords:
    cell = s2cell.lat_lon_to_cell_id(lon, lat)
    cell = pys2.S2CellId(cell)
    #point = pys2.S2LatLng.FromDegrees(lat, lon)
    #cell = pys2.S2CellId(point)
    if cell.face() not in locations_all:
        locations_all[cell.face()] = [cell]
    else:
        locations_all[cell.face()].append(cell)

def split_until_threshold(init_cell, locations, threshold, minimum_threshold):
    if len(locations) == 0:
        return []
    if len(locations) <= minimum_threshold and not init_cell.is_leaf():
        cells = []
        for i in range(4):
            child = init_cell.child(i)
            locations_in_child = []
            for location in locations:
                if child.contains(location):
                    locations_in_child.append(location)

            if len(locations) == len(locations_in_child):
                return split_until_threshold(child, locations_in_child, threshold, minimum_threshold)

        return [init_cell]
    if len(locations) <= threshold:
        return [init_cell]
    cells = []
    for i in range(4):
        child = init_cell.child(i)
        locations_in_child = []
        for location in locations:
            if child.contains(location):
                locations_in_child.append(location)
        out_cells = split_until_threshold(child, locations_in_child, threshold, minimum_threshold)
        cells += out_cells
    return cells

def split_earth_cells():
    cell_ids = []

    for face in range(6):
        locations_on_face = locations_all[face]
        face_cell = pys2.S2CellId.FromFacePosLevel(face, 0, 0)
        cell_ids += split_until_threshold(face_cell, locations_on_face, 50, 5)

    return cell_ids

def get_classes():
    cells = split_earth_cells()
    print(f"Splitted into {len(cells)} classes")

    points = []
    for index, cell_id in enumerate(cells):
        lat, lon = s2cell.cell_id_to_lat_lon(cell_id.id())

        points.append((lon, lat))
    return list(enumerate(points))


def main():
    points = get_classes()
    points = [Point(i[1][0], i[1][1]) for i in points]
    gdf = gpd.GeoDataFrame(geometry=points)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    points2 = [Point(coord[0], coord[1]) for coord in coords]
    gdf2 = gpd.GeoDataFrame(geometry=points2)

    # Plot the polygons on a map
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.stock_img()
    world.plot(ax=ax, color='white', edgecolor='black')
    gdf.plot(ax=ax, marker=".", edgecolor="green", markersize=0.6)
    #gdf2.plot(ax=ax, marker=",", edgecolor="purple", markersize=0.6)
    plt.show()

if __name__ == "__main__":
    main()
