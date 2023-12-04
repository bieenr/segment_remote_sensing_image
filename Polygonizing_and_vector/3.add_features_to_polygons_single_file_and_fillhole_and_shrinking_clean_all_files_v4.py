# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:44:27 2023

@author: ngocd
"""
from osgeo import ogr
from shapely import wkt
from shapely.geometry import Polygon
import glob, sys, os

def main(argv):
    pathIn = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_poly/"
    pathOut = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_poly/Processed2/"

    if(os.path.exists(pathOut) == False):
        os.mkdir(pathOut)

    row = [str(item).zfill(3) for item in map(str, range(238,255))]
    col = [str(item).zfill(3) for item in map(str, range(313,327))]
    #product = ["BldgMask", "RoadMask", "TreeMask", "WaterMask"]  
    for i in row:
        for j in col:
            ltest = glob.glob(pathIn + "/Classification_Building_"+"*("+i+")*("+j+")*.shp")
            if len(ltest)>0:   
                print(ltest[0])
                path1 = ltest[0]
                fileN = os.path.basename(ltest[0])
                fileN_s = fileN[:-4]
                path2 = pathOut + '/' + fileN_s + "_processed.shp"
                
                driver = ogr.GetDriverByName("ESRI Shapefile")
                shapefile = driver.Open(path1)
                #shapefile2 = driver.Open(path2, 1)

                # Get the layer
                layer = shapefile.GetLayer()
                spatialRef = layer.GetSpatialRef()

                dst_layername = "Building areas"
                #drv = ogr.GetDriverByName("ESRI Shapefile")
                shapefile2 = driver.CreateDataSource( path2 )
                layer2 = shapefile2.CreateLayer(dst_layername, srs = spatialRef )

                new_field = ogr.FieldDefn("Area", ogr.OFTReal)
                new_field.SetWidth(32)
                new_field.SetPrecision(4) #added line to set precision
                layer2.CreateField(new_field)

                new_field = ogr.FieldDefn("Vertices", ogr.OFTInteger)
                new_field.SetWidth(6)
                #new_field.SetPrecision(4) #added line to set precision
                layer2.CreateField(new_field)

                new_field = ogr.FieldDefn("Type", ogr.OFTInteger)
                new_field.SetWidth(6)
                #new_field.SetPrecision(4) #added line to set precision
                layer2.CreateField(new_field)   

                # Iterate over the features in the layer
                featureDefn = layer2.GetLayerDefn()

                for feature in layer:
                    # Get the attributes of the feature
                    #attributes = feature.items()
                    geom = feature.GetGeometryRef()
                    #ring = geom.GetGeometryRef(0)
                    str_wkt = geom.ExportToWkt()
                    
                    #Create the shapely polygon and fill holes of the polygon if existed
                    polygon = wkt.loads(str_wkt)
                    new_polygon = Polygon(polygon.exterior.coords)
                    
                    #Shrink the polygon a bit
                    buff_poly = new_polygon.buffer(-0.4)
                    s = str(buff_poly)
                    #print(s)
                    if s != 'POLYGON EMPTY':
                        #Simplify the polygon        
                        simply_poly = buff_poly.simplify(0.4, preserve_topology = False)
                        s2 = str(simply_poly)
                        if s2 != 'POLYGON EMPTY':
                            str_wkt2 = simply_poly.wkt
                            geom2 = ogr.CreateGeometryFromWkt(str_wkt2)
                            #If shrinking
                            if geom2.GetGeometryName() == 'MULTIPOLYGON':
                                for polygon in simply_poly.geoms:
                                    str_wkt3 = polygon.wkt
                                    geom3 = ogr.CreateGeometryFromWkt(str_wkt3)
                                    ring = geom3.GetGeometryRef(0)
                                    feature1 = ogr.Feature(featureDefn)
                                    feature1.SetGeometry(geom3)
                                    
                                    #Count the number of vertices
                                    count1 = ring.GetPointCount()

                                    #Calculate the area of the polygon    
                                    area = geom3.GetArea()

                                    #Update the information of the tables
                                    feature1.SetField("Area", area)    
                                    feature1.SetField("Vertices", count1)
                                    
                                    #An algorithm example to eetermine the value of Type based on area and the number of vertices, should rewrite as a function
                                    type1 = 4
                                    if area < 30:
                                        type1 = 1
                                    elif area > 300:
                                        type1 = 3
                                    elif area < 150:
                                        type1 = 2
                                    elif count1 > 10:
                                        type1 = 3
                                    else:
                                        type1 = 2
                                    feature1.SetField("Type", type1)
                                    #Set new information of the feature to the file
                                    #layer.SetFeature(feature)
                                    layer2.CreateFeature(feature1)
                            else:
                                ring = geom2.GetGeometryRef(0)

                                #Create the new feature and add to the shapefile    
                                feature1 = ogr.Feature(featureDefn)
                                feature1.SetGeometry(geom2)
                                
                                #Count the number of vertices
                                count1 = ring.GetPointCount()

                                #Calculate the area of the polygon    
                                area = geom2.GetArea()

                                #Update the information of the tables
                                feature1.SetField("Area", area)    
                                feature1.SetField("Vertices", count1)
                                
                                #An algorithm example to eetermine the value of Type based on area and the number of vertices, should rewrite as a function
                                type1 = 4
                                if area < 30:
                                    type1 = 1
                                elif area > 300:
                                    type1 = 3
                                elif area < 150:
                                    type1 = 2
                                elif count1 > 10:
                                    type1 = 3
                                else:
                                    type1 = 2
                                feature1.SetField("Type", type1)
                                #Set new information of the feature to the file
                                #layer.SetFeature(feature)
                                layer2.CreateFeature(feature1)
                    
                 #close the shapefile   
                shapefile = None 
                shapefile2 = None                  

if __name__ == "__main__":
    main(sys.argv[1:])
    