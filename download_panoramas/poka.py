import os
import re
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Set the path to the "downloads" folder
path = "./downloads"

# Define a regular expression to match the latitude and longitude in the filename
regex = r"([-]?\d+\.\d+)_([-]?\d+\.\d+)\.jpg$"

# Create an empty list to store the coordinates
coords = []

# Loop over the files in the "downloads" folder
for filename in os.listdir(path):
    # Check if the file is a JPG image
    if filename.endswith(".jpg"):
        # Extract the latitude and longitude from the filename
        match = re.search(regex, filename)
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))
            coords.append((lat, lon))

# Create a list of Shapely Point objects from the coordinates
points = [Point(coord) for coord in coords]

# Create a GeoDataFrame from the points
crs = {'init': 'epsg:4326'}
gdf = gpd.GeoDataFrame(geometry=points, crs=crs)

# Load a shapefile of the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the world map and the points on the same axis
fig, ax = plt.subplots()
world.plot(ax=ax, color='white', edgecolor='black')
gdf.plot(ax=ax, marker='o', color='red', markersize=10)
plt.show()

