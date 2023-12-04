# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sys, datetime, os, glob
import fiona, rasterio, ntpath
import rasterio.mask as mp
import numpy as np
from osgeo import gdal, ogr, osr

#pathTest = 'D:/NAVER/HUST_v1.1/Raster_BldgMask/BldgMask_Row(242)_Col(318).tif'
#pathTest2 = 'D:/NAVER/HUST_v1.1/Results/Luan/infer_results2/Ortho_Row(241)_Col(321).tif'
def main(argv):
    #endDate = '20150101'
    #startDate = '20100101'
    pathIn = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_geom/"
    pathOut = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_poly/"
    
    if(os.path.exists(pathOut) == False):
        os.mkdir(pathOut)

    row = [str(item).zfill(3) for item in map(str, range(238,255))]
    col = [str(item).zfill(3) for item in map(str, range(313,327))]
    #product = ["BldgMask", "RoadMask", "TreeMask", "WaterMask"]  
    for i in row:
        for j in col:
            ltest = glob.glob(pathIn + "/Classification_Building_"+"*("+i+")*("+j+")*.tif")
            if len(ltest)>0:   
                print(ltest[0])
                src_ds = gdal.Open(ltest[0])
                arr = src_ds.GetRasterBand(1)
                prj=src_ds.GetProjection()
                srs1=osr.SpatialReference(wkt=prj)
                if src_ds is None:
                    print('Unable to open %s' % ltest[0])
                    sys.exit(1)

                try:
                    srcband = src_ds.GetRasterBand(1)
                except RuntimeError:
                    # for example, try GetRasterBand(10)
                    print('Band ( %i ) not found' % srcband)
                    #print e
                    sys.exit(1)
                fileN = os.path.basename(ltest[0])
                fileN_s = fileN[:-4]
                path2 = pathOut + '/' + fileN_s + ".shp"
                dst_layername = "Building areas"
                drv = ogr.GetDriverByName("ESRI Shapefile")
                dst_ds = drv.CreateDataSource( path2 )
                dst_layer = dst_ds.CreateLayer(dst_layername, srs = srs1 )
                newField = ogr.FieldDefn('Value', ogr.OFTInteger)
                dst_layer.CreateField(newField)

                gdal.Polygonize( srcband, arr, dst_layer, 0, [], callback=None )                

if __name__ == "__main__":
    main(sys.argv[1:])

