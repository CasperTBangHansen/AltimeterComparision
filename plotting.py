from pathlib import Path
from typing import List, Tuple
from datetime import date
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import warnings

def make_png(
        image: npt.ArrayLike,
        figsize: Tuple[int, int],
        output_path: Path,
        extent: List[int] = None,
        vmin: float = None,
        vmax: float = None,
        cmap: str = 'jet',
        title: str = None,
        cbar_label: str = None,
        fontsize: int = 10,
        ticksize: int = 10,
        titlesize: int = 10
    ):
    """Makes a png based of a grid"""
    if extent is None:
        extent = [-180, 180, -90, 90]
    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    im = ax.imshow(image, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=ticksize)

    ax.coastlines()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.set_xlabel(f"Longitude [\N{DEGREE SIGN}]", fontsize=fontsize)
    ax.set_ylabel(f"Latitude [\N{DEGREE SIGN}]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)

    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    
    plt.savefig(output_path, format='png')
    plt.close()


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or averaging.
    Number of output dimensions must match number of input dimensions.
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(len(new_shape)):
            if operation.lower() == "sum":
                ndarray = np.nansum(ndarray, -1*(i+1) )
            elif operation.lower() in ["mean", "average", "avg"]:
                ndarray = np.nanmean(ndarray, -1*(i+1), )
    return ndarray

def rotate_data(data: xr.DataArray, coord: str, upper_boundary: float):
    # Assign lower bound
    bool_arr = data[coord] < upper_boundary
    if bool_arr.sum().item() == len(data[coord]):
        return data.isel(**{coord: bool_arr}).values
    new_data = np.empty(data.shape)
    new_data[:, ~bool_arr] = data.isel(**{coord: bool_arr})
    new_data[:, bool_arr] = data.isel(**{coord: ~bool_arr})
    return new_data


def cmems_to_date(file: Path) -> date:
    cleaned_filename = file.name.replace('.nc', '').replace('.png', '').replace('dt_global_allsat_phy_l4_', '')
    if 'vDT2021' in cleaned_filename:
        date_name = cleaned_filename.split('_')[-2]
    else:
        date_name = cleaned_filename.split('_')[0]
    return date(int(date_name[:4]), int(date_name[4:6]), int(date_name[6:]))

def measures_to_date(file: Path) -> date:
    date_name = file.name.replace('.nc', '').replace('.png', '').replace('ssh_grids_v2205_', '')
    return date(int(date_name[:4]), int(date_name[4:6]), int(date_name[6:8]))

def processed_to_date(file: Path) -> date:
    year, month, day = file.name.replace('.nc', '').replace('.png', '').split('_')
    return date(int(year), int(month), int(day))

def date_to_processed(file_date: date, extension: str = 'nc') -> Path:
    return Path(f"{file_date.year}_{file_date.month}_{file_date.day}.{extension}")