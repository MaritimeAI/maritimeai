from typing import Any, Callable

import cv2 as cv
import numpy as np

from osgeo import gdal, ogr, gdalconst
gdal.UseExceptions()

from .const import COLORMAP
from .utils import adjust_gamma
from .utils import apply_kmeans
from .utils import extend_mask


# TODO: channels: Grayscale + Alpha, CMYK, etc.
CHANNELS_RGB = (
        gdal.GCI_RedBand,
        gdal.GCI_GreenBand,
        gdal.GCI_BlueBand,
)

CHANNELS_RGBA = CHANNELS_RGB + (
        gdal.GCI_AlphaBand,
)


def read_to_channels(filename: str, channels: Any = None) -> gdal.Dataset:
    if not isinstance(channels, (tuple, list, dict, set)):
        channels = CHANNELS_RGB
    num_channels = len(channels) + 1  # TODO: assert for zero
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset.RasterCount < 1:
        del dataset
        print(f"ERROR: dataset {filename} has no rasters!")
        return None
    # Add channels (Grayscale -> RGB[A])
    if dataset.RasterCount < num_channels:
        tempset = gdal.GetDriverByName('MEM').CreateCopy('', dataset, 0)
        del dataset  # release resources
        band = tempset.GetRasterBand(1)
        meta = band.GetMetadata()
        layer = band.ReadAsArray()
        band.SetColorInterpretation(channels[0])
        for i, c in zip(range(tempset.RasterCount + 1, num_channels + 1),
                        channels[1:]):
            tempset.AddBand()
            band = tempset.GetRasterBand(i)
            band.WriteArray(layer)
            band.SetColorInterpretation(c)
            band.SetMetadata(meta)
        del band
    return tempset

def enhance_dataset(source: str, mask_pad: int = 0,
                    gamma: float = 1.0,
                    bilateral: tuple = None,
                    size_tile: int = None,
                    size_cache: int = None,
                    tilewise: bool = True) -> gdal.Dataset:
    if size_cache is None:
        size_cache = 10 * 1024 * 1024  # 10 MiB
    # TODO: support gda.Dataset for 'source' (not open)
    dataset = gdal.Open(source, gdal.GA_Update)
    if dataset.RasterCount < 1:
        del dataset
        print("ERROR: dataset {source} has no rasters!")
        return None
    print(f"GeoTIFF rasters = {dataset.RasterCount}")
    print(f"GeoTIFF raster shape = ({dataset.RasterYSize}, "
          f"{dataset.RasterXSize})")
    if tilewise:
        # Processing tile-wise
        if not isinstance(size_tile, int) or size_tile < 0:
            size_tile = 256
        elif size_tile > 3072:
            size_tile = 3072  # tiles over 3200 px seem not to work well
        size_tile_x = size_tile
        size_tile_y = size_tile
        tiles = dataset.GetTiledVirtualMemArray(eAccess=gdalconst.GF_Write,
                                                tilexsize=size_tile_x,
                                                tileysize=size_tile_y,
                                                cache_size=size_cache,
                                                tile_organization=gdalconst.GTO_TIP)
        try:
            print(f"Tiles array shape = {tiles.shape}",
                  "(tilesY, tilesX, Y, X, channels)")
            num_channels = tiles.shape[-1]
            print("Enhancing...")
            for i in range(tiles.shape[0]):
                for j in range(tiles.shape[1]):
                    # TODO: if propagate_first:
                    image = tiles[i, j, ..., 0]  # the first tile's channel
                    if mask_pad:
                        image = extend_mask(image, mask_pad)
                    if gamma != 1.0:
                        image = adjust_gamma(image, gamma=gamma, pad=0)
                    if isinstance(bilateral, (tuple, list)) and bilateral:
                        image = cv.bilateralFilter(image, *bilateral)
                    # TODO: if propagate_first:
                    tiles[i, j, ...] = np.repeat(image[..., None], num_channels,
                                                 axis=-1)
                    print('.', end='', flush=True)
                print()
        finally:
            del tiles
    else:
        raise NotImplemented()
    dataset.FlushCache()
    return dataset

def save_grayscale(source: Any, target: Any, band: int = 1,
                   callback: Callable = None) -> None:
    print(f"Saving as grayscale to {target}...")
    gdal.Translate(target, source, options=['-b', str(band),
                                            '-colorinterp', 'gray',
                                            '-co', 'COMPRESS=DEFLATE'],
                   callback=callback)
    return None

def cluster_dataset(source: gdal.Dataset, clusters: int = 7, plt: Any = None,
                    show_histogram: bool = False,
                    colormap: Any = None) -> gdal.Dataset:
    # TODO: check if source opened with GA_Update or handle exceptions
    if not isinstance(source, gdal.Dataset):
        print("ERROR: clustering expects GDAL Dataset as input!")
        return source
    if colormap is None:
        colormap = [[0, 0, 0, 255]] + COLORMAP[:-1]  # prepend with black
    if not isinstance(colormap, (np.ndarray, tuple, list)):
        print("ERROR: clustering expects colormap to be 2D array or list!")
        return source
    elif isinstance(colormap, np.ndarray) and colormap.ndim < 2:
        print("ERROR: clustering expects colormap to be 2D array or list!")
        return source
    elif isinstance(colormap, (list, tuple)):
        colormap = np.array(colormap, dtype=np.uint8)
    image = source.ReadAsArray()
    try:
        print(f"Clustering {image.shape} image...")
        grayscale = image[0, ...]
        if plt is not None:
            plt.subplots(1, 1, figsize=(25, 10))[1].imshow(grayscale,
                                                           cmap='gray')
            if show_histogram:
                histogram = cv.calcHist([np.moveaxis(image, 0, -1)], [0], None,
                                        [127], [1, 255])
                histograms = plt.subplots(1, 1, figsize=(25, 10))[1]
                histograms.hist(histogram, [127], [1, 255])
                histograms.set_xlim([0, 127])
        clustered = apply_kmeans(grayscale, clusters, eps=0.95)
        print("Done!")

        # Assemble into channels with colormap applied
        for i, c in zip(range(source.RasterCount), ('R', 'G', 'B', 'A')):
            image[i, ...] = (cv.LUT(clustered, colormap[..., i])
                             .astype(np.uint8) |
                             (grayscale == 255).astype(np.uint8) * 255)
            source.GetRasterBand(i + 1).WriteArray(image[i, ...])
        source.FlushCache()
        if plt is not None:
            image = np.moveaxis(image, 0, -1)
            plt.subplots(1, 1, figsize=(25, 10))[1].imshow(image)
            plt.show()
            if show_histogram:
                image = np.moveaxis(source.ReadAsArray(), 0, -1)
                print("Building histograms...")
                histograms = plt.subplots(1, 1, figsize=(25, 10))[1]
                for i, c in enumerate(['b', 'g', 'r']):
                    histogram = cv.calcHist([image], [i], None, [127], [1, 255])
                    histograms.plot(histogram, color=c)
                    histograms.set_xlim([0, 127])
                plt.show()
    finally:
        del image
    return source

def vectorize_dataset(source: gdal.Dataset, target: str,
                      callback: Callable = None) -> None:
    if not isinstance(source, gdal.Dataset):
        print("ERROR: vectorization expects GDAL Dataset as a source!")
    tempset = (ogr.GetDriverByName('ESRI Shapefile')
               .CreateDataSource(target))
    try:
        band_source = source.GetRasterBand(1)
        band_mask = band_source.GetMaskBand()
        srs = source.GetSpatialRef()  # requires GDAL >= 3.0.4

        layer = tempset.CreateLayer('out', geom_type=ogr.wkbPolygon, srs=srs)
        layer.CreateField(ogr.FieldDefn('DN', ogr.OFTInteger))

        options = []
        field = 0

        print(f"Saving shape to {target}...")
        gdal.Polygonize(band_source, band_mask, layer, field, options,
                        callback=callback)
    finally:
        del tempset
    return None
