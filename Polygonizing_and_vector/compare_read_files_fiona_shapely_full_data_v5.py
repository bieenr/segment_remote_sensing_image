# -*- coding: utf-8 -*-
import fiona
from shapely.geometry import shape
import shapely, glob, sys, csv
from rtree import index
from collections import OrderedDict
import os
# /mnt/data/bientd/segment_building/mask_polygon/infer_results4_fixed_poly/Processed2
def main(argv):
    pathIn1 = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_poly/Processed2/"
    pathIn2 = "/mnt/data/bientd/segment_building/mask_polygon/infer_results4_fixed_poly/Processed2"
    pathOut = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274/infer_results4_fixed_poly/Processed2/matching_predict_groundtruth_v4.csv"

 

    row = [str(item).zfill(3) for item in map(str, range(238,255))]
    col = [str(item).zfill(3) for item in map(str, range(313,327))]
    #product = ["BldgMask", "RoadMask", "TreeMask", "WaterMask"]  
    
    fieldN = []
    fieldN.append(('Scene', 'None'))
    fieldN.append(('Small', 'None'))
    fieldN.append(('Simple', 'None'))
    fieldN.append(('Complex', 'None'))
    fieldN.append(('Small_correct', 'None'))
    fieldN.append(('Simple_correct', 'None'))
    fieldN.append(('Complex_correct', 'None'))
    fieldN.append(('Small_percentage', 'None'))
    fieldN.append(('Simple_percentage', 'None'))
    fieldN.append(('Complex_percentage', 'None'))
    
    fieldN = OrderedDict(fieldN)
    fList=open(pathOut, 'w', newline='')
    writer = csv.DictWriter(fList, fieldnames=fieldN)
    writer.writeheader()
    for i in row:
        for j in col:
            #print(i)
            l1 = glob.glob(pathIn1 + "/Classification_Building_"+"*("+i+")*("+j+")*.shp")
            l2 = glob.glob(pathIn2 + "/Mask_Building_"+"*("+i+")*("+j+")*.shp")
            if len(l1)>0 and len(l2)>0:   
                print(i+" "+j)
                path1 = l1[0]
                path2 = l2[0]
                shapefile1 = fiona.open(path1) 
                shapefile2 = fiona.open(path2)
                typeCount_predict=[0,0,0]
                for feat1 in shapefile1:
                    if feat1['properties']['Type']==1:
                        typeCount_predict[0] = typeCount_predict[0] + 1
                    elif feat1['properties']['Type']==2:
                        typeCount_predict[1] = typeCount_predict[1] + 1
                    else:
                        typeCount_predict[2] = typeCount_predict[2] + 1
                
                fc_intersect = []

                idx = index.Index()
                for k,feat1 in enumerate(shapefile1):
                    idx.insert(k, shape(feat1['geometry']).bounds)

                count1 = 0
                for feat2 in shapefile2:
                    # Test for potential intersection with each feature of the other feature collection
                    for intersect_maybe in idx.intersection(shape(feat2['geometry']).bounds):
                        # Confirm intersection
                        if shape(feat2['geometry']).intersects(shape(shapefile1[intersect_maybe]['geometry'])):
                            a = shape(feat2['geometry']).intersection(shape(shapefile1[intersect_maybe]['geometry']))
                            if a.area/shape(feat2['geometry']).area >= 0.5 and a.area/shape(shapefile1[intersect_maybe]['geometry']).area >= 0.5:
                                fc_intersect.append([shapefile1[intersect_maybe], feat2])
                                count1 = count1 + 1
                
                typeCount=[0,0,0]
                for feat3 in fc_intersect:
                    a1 = feat3[0]
                    if a1['properties']['Type']==1:
                        typeCount[0] = typeCount[0] + 1
                    elif a1['properties']['Type']==2:
                        typeCount[1] = typeCount[1] + 1
                    else:
                        typeCount[2] = typeCount[2] + 1
                
                rowWrite = {}
                rowWrite['Scene'] = 'Row_' + str(i) + '_Col_' + str(j)
                rowWrite['Small'] = typeCount_predict[0]
                
                rowWrite['Simple'] = typeCount_predict[1]
                rowWrite['Complex'] = typeCount_predict[2]
                rowWrite['Small_correct'] = typeCount[0]
                rowWrite['Simple_correct'] = typeCount[1]
                rowWrite['Complex_correct'] = typeCount[2]
                if typeCount_predict[0] != 0 :
                    rowWrite['Small_percentage'] = typeCount[0]/typeCount_predict[0]*100
                else :
                    rowWrite['Small_percentage'] = 0

                if typeCount_predict[1] != 0 :
                    rowWrite['Simple_percentage'] = typeCount[1]/typeCount_predict[1]*100
                else :
                    rowWrite['Simple_percentage'] = 0
                
                if typeCount_predict[2] != 0 :
                    rowWrite['Complex_percentage'] = typeCount[2]/typeCount_predict[2]*100
                else :
                    rowWrite['Complex_percentage'] = 0

            
                writer.writerow(rowWrite)
                
                
if __name__ == "__main__":
    main(sys.argv[1:])

'''
path2 = "D:/NAVER/HUST_v1.2/Mask2_Building_nodata_poly/Processed2/Mask_Building_Row(242)_Col(319)_processed.shp"
path1 = "D:/NAVER/HUST_v1.1/Results/Luan/infer_results4_fixed_poly/Processed2/Classification_Building_Row(242)_Col(319)_processed.shp"
# Open the shapefile
shapefile1 = fiona.open(path1) 
shapefile2 = fiona.open(path2)

print(len(shapefile1))
print(shapefile2[0])
#a = shapefile1[0]
#print(a)
print(len(shapefile1))
typeCount_predict=[0,0,0]
for feat1 in shapefile1:
    if feat1['properties']['Type']==1:
        typeCount_predict[0] = typeCount_predict[0] + 1
    elif feat1['properties']['Type']==2:
        typeCount_predict[1] = typeCount_predict[1] + 1
    else:
        typeCount_predict[2] = typeCount_predict[2] + 1
#a = shapefile2[0]
#print(a)
print(typeCount_predict)

fc_intersect = []

idx = index.Index()
for i,feat1 in enumerate(shapefile1):
    idx.insert(i, shape(feat1['geometry']).bounds)

count1 = 0
for feat2 in shapefile2:
    # Test for potential intersection with each feature of the other feature collection
    for intersect_maybe in idx.intersection(shape(feat2['geometry']).bounds):
        # Confirm intersection
        if shape(feat2['geometry']).intersects(shape(shapefile1[intersect_maybe]['geometry'])):
            a = shape(feat2['geometry']).intersection(shape(shapefile1[intersect_maybe]['geometry']))
            if a.area/shape(feat2['geometry']).area >= 0.5 and a.area/shape(shapefile1[intersect_maybe]['geometry']).area >= 0.5:
                fc_intersect.append([shapefile1[intersect_maybe], feat2])
                count1 = count1 + 1
                #if (count1 % 100 == 1):
                #   print(count1)
                
print(len(fc_intersect))
#print(fc_intersect[0])
#print(fc_intersect[0][0])
#print(fc_intersect[0][1])
print(fc_intersect[0][0])
typeCount=[0,0,0]
for i in fc_intersect:
    a1 = i[0]
    if a1['properties']['Type']==1:
        typeCount[0] = typeCount[0] + 1
    elif a1['properties']['Type']==2:
        typeCount[1] = typeCount[1] + 1
    else:
        typeCount[2] = typeCount[2] + 1

print(typeCount)
#print(typeCount_gt)

#print(a['properties'])
#geometry = shape(a['geometry'])
#print(geometry)
# Iterate over the records
#for record in shapefile:
    # Get the geometry from the record
#    geometry = shape(record['geometry'])
    
    # Print the area of the geometry
#    print(geometry.area)
'''