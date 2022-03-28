from osgeo import gdal, osr, ogr


class GDAL_shp_Data(object):
    def __init__(self, shp_path):
        self.shp_path = shp_path
        self.shp_file_create()

    def shp_file_create(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ogr.RegisterAll()
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # 打开输出文件及图层
        # 输出模板shp 包含待写入的字段信息
        self.outds = driver.CreateDataSource(self.shp_path)
        # 创建空间参考
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        # 创建图层
        self.out_layer = self.outds.CreateLayer("out_polygon", srs, ogr.wkbPolygon)
        field_name = ogr.FieldDefn("scores", ogr.OFTReal)
        self.out_layer.CreateField(field_name)

    def set_shapefile_data(self, polygons, scores):
        for i in range(len(scores)):
            wkt = polygons[i].wkt  # 创建wkt文本点
            temp_geom = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(self.out_layer.GetLayerDefn())  # 创建特征
            feature.SetField("scores", scores[i])
            feature.SetGeometry(temp_geom)
            self.out_layer.CreateFeature(feature)
        self.finish_io()

    def finish_io(self):
        del self.outds
