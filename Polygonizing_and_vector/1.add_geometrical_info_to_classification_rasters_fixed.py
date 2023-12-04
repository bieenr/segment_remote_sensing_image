# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sys, datetime, os, glob
import fiona, rasterio, ntpath
import rasterio.mask as mp
import numpy as np


#pathTest = 'D:/NAVER/HUST_v1.1/Raster_BldgMask/BldgMask_Row(242)_Col(318).tif'
#pathTest2 = 'D:/NAVER/HUST_v1.1/Results/Luan/infer_results2/Ortho_Row(241)_Col(321).tif'
def main(argv):
    endDate = '20150101'
    startDate = '20100101'
    
    pathIn = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274"
    pathOut = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_geom/"
    pathGeom = "/mnt/data/RasterMask_v11/Mask2_Building/"
    
    if(os.path.exists(pathOut) == False):
        os.mkdir(pathOut)
 
    row = [str(item).zfill(3) for item in map(str, range(238,255))]
    col = [str(item).zfill(3) for item in map(str, range(313,327))]
    #product = ["BldgMask", "RoadMask", "TreeMask", "WaterMask"]  
    for i in row:
        for j in col:
            ltest = glob.glob(pathIn + "/Ortho_"+"*("+i+")*("+j+")*.tif")
            
            # ltest = glob.glob(pathIn + "/Mask_Building_"+"*("+i+")*("+j+")*.tif")
            ltest2 = glob.glob(pathGeom + "/Mask_Building_"+"*("+i+")*("+j+")*.tif")
            if len(ltest)>0 and  len(ltest2)>0:   
                arr = rasterio.open(ltest[0]).read(1)
                src = rasterio.open(ltest2[0])
                prof = src.meta.copy()                
                #arr = np.zeros((prof['height'], prof['width']))
                    
                # prof.update({"driver": "GTiff", 
                #              "transform": out_trans, 
                #              "height": cropRaster.shape[1], 
                #              "width": cropRaster.shape[2], 
                #              #'count':1, 
                #              'dtype':rasterio.float32, 
                #              'compress':'lzw',
                #              'nodata':-1000})
                prof.update({"driver": "GTiff",              
                             'compress':'lzw'})
                
                writeFN = "Classification_Building_Row(" + i + ")_Col(" + j + ").tif"
                print(writeFN)
                with rasterio.open(pathOut + '/' + writeFN, "w", **prof) as dest:
                    dest.write(arr.astype(np.uint8), 1)

    #startFileName = 'MOD13A1_forest_Hansen_intact_' + tile
    #pathIn1 = pathWholeTile + QA[0]
#    print(tileSite)
#    for i in tileList:
#        siteExtract = tileSite[i]
#        ID = tileSiteID[i]
#        #windowRange1 = 10
#        #nsite = len(tileSite[i])
#        for j in QA:
#            a = sf1.crop_MOD13A1_Points_ID(pathWholeTile, pathResult, j, band, listRawDirs, ID, i, siteExtract, window=10, mode=2)
#            print(a)
if __name__ == "__main__":
    main(sys.argv[1:])

