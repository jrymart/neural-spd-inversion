import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

def slope(dem_array):
    dz_dy, dz_dx = np.gradient(dem_array)
    slope_array = np.sqrt(dz_dx**2 + dz_dy**2)
    return slope_array

def curvature(dem_array):
    dz_dy, dz_dx = np.gradient(dem_array)
    dz_xy, dz_xx = np.gradient(dz_dx)
    dz_yy, dz_yx = np.gradient(dz_dy)
    laplacian_array = dz_xx + dz_yy
    return laplacian_array

def flow_accumulation(dem_array, cell_size, flow_method):
    grid = RasterModelGrid(dem_array.shape, cell_size)
    grid.add_field("node", "topographic__elevation", dem_array, units="m")
    accumulator = FlowAccumulator(grid, flow_director=flow_method)
    accumulator.run_one_step()
    flow_acc_array = grid.at_node["drainage_area"].reshape(grid.shape)
    return flow_acc_array

def noise_dem(dem_array, noise_level):
    noise = np.random.normal(0, noise_level, dem_array.shape)
    noisy_dem = dem_array + noise
    return noisy_dem
