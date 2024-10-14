#I think that perhaps with a spatial intersection function between the two SHPs 
#(Original and the one that marks the modifications) it could be a quick solution
 
# Vocab: 
# Layer = contents of a .shp file
# Feature = single polygon within a layer
# Field = attributes of a feature (stored in .dbf, but can be accessed by opening .shp in GIS software)

from osgeo import ogr
 
def intersect_shapefiles(input_shapefile1, input_shapefile2, output_shapefile):

    """
    Intersects two shapefiles and writes the result to a new shapefile.
    Args:
        input_shapefile1: Path to the first input shapefile.
        input_shapefile2: Path to the second input shapefile.
        output_shapefile: Path to the output shapefile.
    """

    # Open input shapefiles
    ds1 = ogr.Open(input_shapefile1)
    lyr1 = ds1.GetLayer()

    ds2 = ogr.Open(input_shapefile2)
    lyr2 = ds2.GetLayer()
 
    # Create output shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")

    if os.path.exists(output_shapefile):
        driver.DeleteDataSource(output_shapefile)

    # Create a new "layer" (.shp file contents), and use the same Spatial Ref and Geometry Type (polygon) as the original file
    out_ds = driver.CreateDataSource(output_shapefile)
    out_lyr = out_ds.CreateLayer("intersection", lyr1.GetSpatialRef(), geom_type=lyr1.GetGeomType())
 
    # Create a new field for the intersection type and add to newly created layer
    field_defn = ogr.FieldDefn("intersection_type", ogr.OFTInteger)
    out_lyr.CreateField(field_defn)
 
    # Iterate through features (polygons) in the first layer (input .shp file #1 in intersect/union)
    for feat1 in lyr1:
        geom1 = feat1.GetGeometryRef()

        # Iterate through features (polygons) in the second layer (input .shp file #2 in intersect/union)
        for feat2 in lyr2:
            geom2 = feat2.GetGeometryRef()
            intersection = geom1.Intersection(geom2)
 
            if intersection.IsValid():
                # Create an output feature (polygon)
                out_feat = ogr.Feature(out_lyr.GetLayerDefn())
                out_feat.SetGeometry(intersection)
                out_feat.SetField("intersection_type", 1)  # 1 for intersection
                out_lyr.CreateFeature(out_feat)
 
    # Close datasources
    ds1 = None
    ds2 = None
    out_ds = None
 
def intersect_and_union_shapefiles(input_shapefile1, input_shapefile2, output_shapefile):
    """
    Intersects two shapefiles, unions the result with the first shapefile, and writes the final result to a new shapefile.
 
    Args:
        input_shapefile1: Path to the first input shapefile.
        input_shapefile2: Path to the second input shapefile.
        output_shapefile: Path to the output shapefile.
    """
 
    # Open input shapefiles
    ds1 = ogr.Open(input_shapefile1)
    lyr1 = ds1.GetLayer()
 
    ds2 = ogr.Open(input_shapefile2)
    lyr2 = ds2.GetLayer()
 
    # Create a temporary shapefile for the intersection
    intersection_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource("temp_intersection.shp")
    # "Layer" = contents of a .shp file
    intersection_lyr = intersection_ds.CreateLayer("intersection", lyr1.GetSpatialRef(), geom_type=lyr1.GetGeomType())
 
    # Perform the intersection
    # "Feature" = polygon, so for each polygon in the first shapefile:
    for feat1 in lyr1:
        geom1 = feat1.GetGeometryRef()
        #Maybe add check for instersection first - test to see what happens if no intersection
        #For each polygon in the second .shp file
        for feat2 in lyr2:
            geom2 = feat2.GetGeometryRef()
            intersection = geom1.Intersection(geom2)
            if intersection.IsValid():
                intersection_feat = ogr.Feature(intersection_lyr.GetLayerDefn())
                intersection_feat.SetGeometry(intersection)
                #Add new "intersection" polygon to new output shapefile
                intersection_lyr.CreateFeature(intersection_feat)
 
    # Create the output shapefile for the union
    union_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(output_shapefile)
    union_lyr = union_ds.CreateLayer("union", lyr1.GetSpatialRef(), geom_type=lyr1.GetGeomType())
 
    # Copy features (polygons) from the first shapefile to the union shapefile
    for feat1 in lyr1:
        union_lyr.CreateFeature(feat1)
 
    # Copy features (polygons) from the intersection shapefile to the union shapefile
    intersection_ds = ogr.Open("temp_intersection.shp")
    intersection_lyr = intersection_ds.GetLayer()
    for feat in intersection_lyr:
        union_lyr.CreateFeature(feat)
 
    # Close datasources
    ds1 = None
    ds2 = None
    intersection_ds = None
    union_ds = None
 
# Example usage
input_shapefile1 = "./NBAL_Shapefile/NBALs_SubC_HMUs_Final.shp"
input_shapefile2 = "./NBAL_Shapefile_24_03_07/NBALs_SubC_HMUs_Final.shp"
output_shapefile = "./test_intersect_union_shapefile.shp"
 

def main():
    intersect_and_union_shapefiles(input_shapefile1, input_shapefile2, output_shapefile)

if __name__ == '__main__':
    main()