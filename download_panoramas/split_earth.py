import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt

# Replace the path below with the actual path to your ne_10m_ocean.shp file
ocean_file = "./ne_10m_ocean.shp"

# Get land and ocean data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ocean = gpd.read_file(ocean_file)

# Remove Antarctica
world = world[world['continent'] != 'Antarctica']

# Define a function to create a grid of squares
def create_square_grid(step):
    grid = []
    for lat in np.arange(-90, 90, step):
        for lon in np.arange(-180, 180, step):
            polygon = Polygon([
                (lon, lat),
                (lon + step, lat),
                (lon + step, lat + step),
                (lon, lat + step),
            ])
            grid.append(polygon)
    return grid

# Set grid size
n_areas = 1000
step = np.sqrt(180 * 360 / n_areas)

# Create grid
grid = create_square_grid(step)
all_grid = gpd.GeoDataFrame(geometry=grid)

# Keep only areas that contain some land
land_grid = gpd.overlay(all_grid, world, how='intersection')

# Remove polygons that are wholly water
land_grid = land_grid[~land_grid.geometry.within(ocean.unary_union)]

# Define the threshold area to remove very small polygons
threshold_area = 0.01 * step * step

# Identify small polygons
small_polygons = land_grid[land_grid.geometry.area < threshold_area]

# Expand small polygons slightly
buffer_distance = 0.1 * step
expanded_small_polygons = small_polygons.geometry.buffer(buffer_distance)

# Merge the expanded small polygons with the remaining polygons
merged_polygons = unary_union(land_grid.geometry.tolist() + expanded_small_polygons.tolist())

# Create a new set of polygons
new_land_areas = list(polygonize(merged_polygons))

# Create a new GeoDataFrame with the updated polygons
new_land_grid = gpd.GeoDataFrame(geometry=new_land_areas)

# Print the number of areas and output the list of polygons
print(f'Number of areas: {len(new_land_grid)}')
new_land_areas = new_land_grid.geometry.tolist()
print(new_land_areas)

# Plot the polygons on the map
fig, ax = plt.subplots(figsize=(12, 8))
world.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
new_land_grid.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5)
plt.title("Land Areas")
plt.show()

