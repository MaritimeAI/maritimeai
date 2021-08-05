import gc
import os
import zipfile
import tempfile

from os import path as osp
from glob import glob

import numpy as np

from osgeo import gdal


def process_sentinel1(filename, path_output, area=None, shapes=None, srs=None,
                      ratio=False, negative=False, raw=False):
    if shapes is None:
        shapes = [False]
    else:
        assert isinstance(shapes, (tuple, list, dict, set)), \
                f"'shapes' must be tuple, list, dict or set!"
    if area is None:
        area = 'default'
    if srs is None:
        srs = 'EPSG:32640'  # default SRS
    names = ['HH', 'HV'] + ['RGB'] * any((ratio, negative))  # + ['INV'] * inv
    title = osp.splitext(osp.basename(filename))[0]
    options_warp = {
            'format': 'GTiff',
            'dstSRS': srs,
            'creationOptions': ['COMPRESS=DEFLATE'],
            'xRes': 40,
            'yRes': 40
    }
    driver_mem = gdal.GetDriverByName('MEM')
    with tempfile.TemporaryDirectory() as path_temp:
        with zipfile.ZipFile(filename, 'r') as archive:
            archive.extractall(path_temp)
            path_safe = glob(osp.join(path_temp, f"*.SAFE"))[0]
            dataset = gdal.Open(path_safe, gdal.GA_ReadOnly)
            subsets = dataset.GetSubDatasets()
            datasets = {}
            for i, p in enumerate(names):
                print(f"Reading {subsets[i][1]}...")
                datasets[p] = gdal.Open(subsets[i][0], gdal.GA_ReadOnly)
            filenames = []
            name_area = osp.splitext(osp.basename(area))[0]
            print(f"Warping polarizations...")
            for name, source in datasets.items():
                #
                # Prepare filenames and paths
                #
                for shape in shapes:
                    if shape:
                        name_shape = osp.basename(shape)
                        name_shape = osp.splitext(name_shape)[0]
                        data_prefix = f"{name_area}_{name_shape}"
                        options_cutline = {'cutlineDSName': shape,
                                           'cropToCutline': True}
                    else:
                        data_prefix = f"{name_area}"
                        options_cutline = {}
                    data_output = osp.join(path_output, data_prefix)
                    if not name in ('RGB', 'INV'):
                        data_output = osp.join(data_output, name.lower())
                        os.makedirs(data_output, exist_ok=True)
                        print(f"{data_output.replace(path_output, '')}")
                        destination = f"{osp.join(data_output, title)}.tiff"
                        filenames.append(destination)
                        if raw:
                            gdal.Warp(destination, source, **options_warp,
                                      **options_cutline)
                        else:
                            memoset = driver_mem.CreateCopy('', source, 0)
                            # Work with in-memory dataset only
                            band = memoset.GetRasterBand(1)
                            image = band.ReadAsArray()
                            mask = image == 0
                            image = np.ma.array(image, mask=mask,
                                                dtype=np.float32)
                            del mask
                            stats = (image.mean().astype(np.float32),
                                     image.std().astype(np.float32))
                            image = np.ma.tanh(image / (stats[0] + 2 *
                                                        stats[1]),
                                               dtype=np.float32)
                            del stats
                            # Convert to byte type
                            image = (image * 254 + 1).astype(np.uint8)
                            # Write channel to the MEM dataset
                            band.WriteArray(image)
                            del image
                            band.SetColorInterpretation(gdal.GCI_GrayIndex)
                            gdal.Warp(destination, memoset, **options_warp,
                                      outputType=gdal.GDT_Byte,
                                      **options_cutline)
                        datasets[name] = None
                        del memoset
                        del source
                    else:
                        # Basic RGB processing
                        memoset = driver_mem.CreateCopy('', source, 0)
                        datasets[name] = None
                        del source
                        gc.collect()
                        # Work with in-memory dataset only
                        band_hh = memoset.GetRasterBand(1)
                        image_hh = band_hh.ReadAsArray()
                        mask_hh = image_hh == 0
                        image_hh = np.ma.array(image_hh, mask=mask_hh,
                                               dtype=np.float32)
                        del mask_hh
                        band_hv = memoset.GetRasterBand(2)
                        image_hv = band_hv.ReadAsArray()
                        mask_hv = image_hv == 0
                        image_hv = np.ma.array(image_hv, mask=mask_hv,
                                               dtype=np.float32)
                        del mask_hv
                        stats_hh = (image_hh.mean().astype(np.float32),
                                    image_hh.std().astype(np.float32))
                        stats_hv = (image_hv.mean().astype(np.float32),
                                    image_hv.std().astype(np.float32))
                        image_hh = np.ma.tanh(image_hh / (stats_hh[0] + 2 *
                                                          stats_hh[1]),
                                              dtype=np.float32)
                        image_hv = np.ma.tanh(image_hv / (stats_hv[0] + 2 *
                                                          stats_hv[1]),
                                              dtype=np.float32)
                        # if not raw:
                            # TODO: save to 8-bit GeoTIFFs
                            # print(f"DEBUG: Yet ok...")
                        if ratio:
                            image_ratio = image_hh / image_hv
                            stats_ratio = (image_ratio.mean().astype(np.float32),
                                           image_ratio.std().astype(np.float32))
                            image_ratio = image_ratio / image_ratio.max()
                        if negative:
                            image_negative = (np.float32(1) -
                                              np.ma.tanh(image_hh / image_hv,
                                                         dtype=np.float32))
                        # Convert to byte type
                        image_hh = (image_hh * 254 + 1).astype(np.uint8)
                        image_hv = (image_hv * 254 + 1).astype(np.uint8)
                        if ratio:
                            image_ratio = (image_ratio * 254 + 1)\
                                          .astype(np.uint8)
                        if negative:
                            image_negative = (image_negative * 254 + 1)\
                                             .astype(np.uint8)
                        # Write channels to the MEM dataset
                        memoset.AddBand()
                        band_ex = memoset.GetRasterBand(3)
                        band_ex.SetColorInterpretation(gdal.GCI_BlueBand)
                        band_hh.WriteArray(image_hh)
                        band_hh.SetColorInterpretation(gdal.GCI_RedBand)
                        band_hv.WriteArray(image_hv)
                        band_hv.SetColorInterpretation(gdal.GCI_GreenBand)
                        # Create ratio band (HH, HV, HH/HV)
                        if ratio:
                            band_ex.WriteArray(image_ratio)
                            band_ex.SetMetadata({'POLARISATION': 'HH/HV',
                                                 'SWATH': 'EW'})
                            path_ratio = osp.join(data_output, 'ratio')
                            os.makedirs(path_ratio, exist_ok=True)
                            print(f"{path_ratio.replace(path_output, '')}")
                            destination = f"{osp.join(path_ratio, title)}.tiff"
                            filenames.append(destination)
                            gdal.Warp(destination, memoset, **options_warp,
                                      outputType=gdal.GDT_Byte,
                                      **options_cutline)
                        # Create negative band (HH, HV, 1 - HH/HV)
                        if negative:
                            band_ex.WriteArray(image_negative)
                            band_ex.SetMetadata({'POLARISATION': '1 - HH/HV',
                                                 'SWATH': 'EW'})
                            path_negative = osp.join(data_output, 'negative')
                            os.makedirs(path_negative, exist_ok=True)
                            print(f"{path_negative.replace(path_output, '')}")
                            destination = f"{osp.join(path_negative, title)}.tiff"
                            filenames.append(destination)
                            gdal.Warp(destination, memoset, **options_warp,
                                      outputType=gdal.GDT_Byte,
                                      **options_cutline)
                        del memoset
    print(f"Done!")
    return filenames


def process_sentinel2(filename, path_output, area=None, shapes=None):
    if shapes is None:
        shapes = [False]
    else:
        assert isinstance(shapes, (tuple, list, dict, set)), \
                f"'shapes' must be tuple, list, dict or set!"
    if area is None:
        area = 'default'
    title = osp.splitext(osp.basename(filename))[0]
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    subsets = dataset.GetSubDatasets()
    assert len(subsets) > 0, f"no sub datasets found!"
    dataset = gdal.Open(subsets[0][0], gdal.GA_ReadOnly)
    # print(f"{snapshot.title} -->")
    print(f"Reading {subsets[0][1][:1].lower()}",
          f"{subsets[0][1][1:]}", sep='')
    image = dataset.ReadAsArray()
    if image.ndim < 3:
        image = image[None, ...]
    image = np.moveaxis(image[:3, ...], 0, -1)  # CHW -> HWC
    image = image.astype(np.float32)
    print(f"Calculating optimal histogram...")
    clip = image.mean().astype(np.float32) * np.float32(2)
    image = image / clip
    image = np.tanh(image)
    # Apply gamma correction here (TODO)
    image = (image * 254 + 1).round().astype(np.uint8)
    # Apply nodata mask here (TODO)
    image = np.moveaxis(image, -1, 0)  # HWC -> CHW
    print(f"Applying histogram...")
    tempset = gdal.GetDriverByName('MEM')\
              .CreateCopy('', dataset, 0)
    for i in range(image.shape[0]):
        band = tempset.GetRasterBand(i + 1)
        band.WriteArray(image[i].astype(np.uint16))
        del band
    print(f"Writing to temporary file...")
    with tempfile.TemporaryDirectory() as path_temp:
        temp = osp.join(path_temp, 'temp.tiff')
        gdal.Translate(temp, tempset,
                       creationOptions=['COMPRESS=DEFLATE'],
                       format='GTiff', bandList=[1, 2, 3],
                       outputType=gdal.GDT_Byte)
        del tempset
        #
        # Prepare filenames and paths
        #
        filenames = []
        name_area = osp.splitext(osp.basename(area))[0]
        for shape in shapes:
            if shape:
                name_shape = osp.basename(shape)
                name_shape = osp.splitext(name_shape)[0]
                data_prefix = f"{name_area}_{name_shape}"
                options = {'cutlineDSName': shape,
                           'cropToCutline': True}
            else:
                data_prefix = f"{name_area}"
                options = {}
            data_output = osp.join(path_output,
                                       # data_name,
                                       data_prefix)
            os.makedirs(data_output, exist_ok=True)
            destination = f"{osp.join(data_output, title)}.tiff"
            filenames.append(destination)
            gdal.Warp(destination, temp, **options)
    print(f"Done!")
    return filenames

