import os, re
import geopandas as gpd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import pickle
import s2cell

try:
    import pywraps2 as pys2
except Exception:
    print("failed to import pywraps2, you can only load stuff from points.pickle")

# Set the path to the "downloads" folder
path = "./download_panoramas/downloads"


# Create an empty list to store the coordinates
coords = []

locations_all = {}

# Loop over the files in the "downloads" folder
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
    if len(locations) <= threshold or init_cell.level() > 7:
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
        cell_ids += split_until_threshold(face_cell, locations_on_face, 105, 10)

    return cell_ids

def get_classes():
    if os.path.isfile("points.pickle"):
        print("Using existing points.pickle")
        with open('points.pickle', 'rb') as handle:
            b = pickle.load(handle)
            return b

    if os.path.isfile("images.txt"):
        print("images.txt exists, using that...")
        with open("images.txt", "r") as f:
            lines = f.readlines()
    else:
        lines = os.listdir(path)


    for filename in lines:
        filename = filename.strip()
        # Check if the file is a JPG image
        if filename.endswith(".jpg"):
            # Extract the latitude and longitude from the filename
            out = filename.split(".jpg")[0]
            lat, lon = out.split("_")
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

    for lat, lon in coords:
        cell = s2cell.lat_lon_to_cell_id(lat, lon)
        cell = pys2.S2CellId(cell)
        #point = pys2.S2LatLng.FromDegrees(lat, lon)
        #cell = pys2.S2CellId(point)
        if cell.face() not in locations_all:
            locations_all[cell.face()] = [cell]
        else:
            locations_all[cell.face()].append(cell)


    cells = split_earth_cells()
    print(f"Splitted into {len(cells)} classes")

    points = []
    for index, cell_id in enumerate(cells):
        lat, lon = s2cell.cell_id_to_lat_lon(cell_id.id())
        assert lat <= 90
        assert lat >= -90

        points.append([lon, lat])
    return list(enumerate(points))


def main():

    points = get_classes()
    with open('points.pickle', 'wb') as handle:
        pickle.dump(points, handle, protocol=4)

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

