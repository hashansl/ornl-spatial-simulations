
# Import Libraries
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry as geom
from scipy.ndimage import gaussian_filter

import libpysal as ps
from esda.moran import Moran

def generate_grid_dataframe(grid_side_length, autocorrelation="positive", random_seed=42):
    """
    Generates a GeoDataFrame based on the grid size of one side with different types of spatial autocorrelation.
    
    Parameters:
        grid_side_length (int): The number of squares along one side of the grid.
        autocorrelation (str): Type of spatial autocorrelation. Options:
                               "none" - No spatial correlation (pure random)
                               "positive" - Spatially smoothed using a Gaussian filter
                               "negative" - Spatially negative correlation (inverted)
                               "cluster" - Clustered spatial pattern with defined centers
        random_seed (int): Random seed for reproducibility.

    Returns:
        GeoDataFrame: A GeoDataFrame containing square geometries and values based on the selected correlation.
    """
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    num_squares = grid_side_length ** 2
    df = pd.DataFrame({'Index': np.arange(num_squares)})

    # Define grid size and statistical parameters
    grid_size = (grid_side_length, grid_side_length)
    mean, std_dev = 0.5, 0.125

    # Generate initial random values
    random_values = np.random.normal(mean, std_dev, grid_size)

    # Generate intitial values from Inverse Gaussian distribution
    # random_values = np.random.wald(mean, std_dev, grid_size)

    if autocorrelation == "none":
        values = random_values  # No spatial correlation

    elif autocorrelation == "positive":
        values = gaussian_filter(random_values, sigma=1.5)  # Apply Gaussian smoothing for spatial correlation

    elif autocorrelation == "negative":
        smoothed_values = gaussian_filter(random_values, sigma=1.5)
        checkerboard = np.indices(grid_size).sum(axis=0) % 2  # Create checkerboard pattern (alternating 0s and 1s)
        values = smoothed_values * (-1) ** checkerboard  # Flip alternating cells # This operation flips the sign of every other cell in a checkerboard pattern.
        # add background noise
        values += np.random.normal(0, 0.05, grid_size)

    elif autocorrelation == "cluster":
        # Generate clustered pattern
        values = np.zeros(grid_size)  # Start with zeros
        # Define cluster centers (example: 2 clusters)
        n_clusters = 2
        cluster_centers = np.random.randint(0, grid_side_length, (n_clusters, 2))
        cluster_radius = grid_side_length // 4  # Define cluster influence radius
        
        # Create distance-based clusters
        for center in cluster_centers:
            y, x = np.ogrid[:grid_size[0], :grid_size[1]]
            # Calculate distance from cluster center
            dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            # Apply Gaussian decay within radius
            cluster_effect = np.exp(-dist_from_center**2 / (2 * cluster_radius**2))
            values += cluster_effect * np.random.normal(0.5, 0.2)  # Add cluster effect
        
        # Add background noise
        values += np.random.normal(0, 0.05, grid_size)

    else:
        raise ValueError("Invalid autocorrelation type. Choose from 'none', 'positive', 'negative', or 'cluster'.")

    # Assign values to the DataFrame
    df['Value'] = values.ravel()

    # Function to calculate square coordinates
    def calculate_square_coordinates(row):
        value = row['Index']
        x = value % grid_side_length
        y = value // grid_side_length
        return geom.Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])

    # Convert to GeoDataFrame with square geometries
    df['geometry'] = df.apply(calculate_square_coordinates, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf


def calculate_moran_i(gdf, grid_side_length):
    """
    Calculates Moran's I for the given GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): A spatial dataframe with values and geometry.
        grid_side_length (int): The number of rows/columns in the square grid.

    Returns:
        float: Moran's I value indicating spatial autocorrelation.
    """
    # Create spatial weights matrix (rook contiguity)
    w = ps.weights.lat2W(grid_side_length, grid_side_length)
    w.transform = 'r'  # Row-standardized weights

    # Extract the 'Value' column for Moran's I computation
    values = gdf['Value'].values

    # Compute Moran’s I
    moran = Moran(values, w)

    return moran.I