# import locate
# from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS

# def getGPSPosition(image: Image.Image, image_locations:list[tuple[float,float]]) -> list[float]:
#     try:
#         img = Image.open(image)
#         width, height = img.size
#         exif_data = img._getexif()

#         gps_info = {}
#         if exif_data and 0x8825 in exif_data:
#             # Extract GPSInfo from exif_data
#             gps_info = {GPSTAGS.get(tag, tag): value for tag, value in exif_data[0x8825].items()}
#             for key in ['GPSLatitude', 'GPSLongitude']:
#                 if key in gps_info:
#                     degrees, minutes, seconds = gps_info[key]
#                     gps_info[key] = degrees + (minutes / 60) + (seconds / 3600)

#         altitude = gps_info.get('GPSAltitude', None)
#         latitude = gps_info.get('GPSLatitude', None)
#         latitude_numerator, latitude_denominator = str((latitude)).split('/')
#         latitude_degree = float(latitude_numerator)/float(latitude_denominator)
#         longitude = gps_info.get('GPSLongitude', None)
#         longitude_numerator, longitude_denominator = str((longitude)).split('/')
#         longitude_degree = float(longitude_numerator)/float(longitude_denominator)

#         fd = open(image, encoding='latin-1')
#         d = fd.read()
#         xmp_start = d.find('<x:xmpmeta')
#         xmp_end = d.find('</x:xmpmeta')
#         xmp_str = d[xmp_start:xmp_end]
#         yaw_start = xmp_str.find('drone-dji:FlightYawDegree="') + len('drone-dji:FlightYawDegree="')
#         yaw_end = xmp_str.find('"', yaw_start)
#         yaw = float(xmp_str[yaw_start:yaw_end])

#         print(f"Image size (width x height): {width} x {height}")
#         print(f"Latitude: {latitude_degree}")
#         print(f"Longitude: {longitude_degree}")
#         print(f"Altitude: {altitude}")
#         print(f'yaw: {yaw}')
#         x_center = image_locations[0][0]
#         y_center = image_locations[0][1]
#         print(f"center: {x_center}, {y_center}")

#         # locate.locate( latitude, longitude, altitude, bearing, cam_fov, imgWidth, imgHeight, obj_x_px, obj_y_px)
#         # cam_fov was consistently given as 73.7 in exif viewer
#         obj_long, obj_lat = locate.locate(latitude_degree, longitude_degree, altitude, yaw, 73.7, width, height, x_center, y_center )
#         print(f"obj_latitude={obj_lat}, obj_longitude={obj_long}")

#     except Exception:
#         return

# if __name__ == "__main__":
#     image = "/Users/dominicgartner/Desktop/SUAV dataset/pict/DJI_0679.JPG"
#     gps_values = getGPSPosition(image, [(0.5, 0.5)])



# import locate
# from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS

# def getGPSPosition(image: Image.Image, image_locations:list[tuple[float,float]]) -> list[float]:
#     try:
#         img = Image.open(image)
#         width, height = img.size
#         exif_data = img._getexif()

#         gps_info = {}
#         if exif_data and 0x8825 in exif_data:
#             # Extract GPSInfo from exif_data
#             gps_info = {GPSTAGS.get(tag, tag): value for tag, value in exif_data[0x8825].items()}
#             for key in ['GPSLatitude', 'GPSLongitude']:
#                 if key in gps_info:
#                     degrees, minutes, seconds = gps_info[key]
#                     gps_info[key] = degrees + (minutes / 60) + (seconds / 3600)

#         # change the altitude to maryland california's altitude. Calgary (1045m), maryland (32m). Difference is 1045 - 32 = 1013
#         altitude = float((gps_info.get('GPSAltitude', None)) - 1013)
#         latitude = gps_info.get('GPSLatitude', None)
#         latitude_numerator, latitude_denominator = str((latitude)).split('/')
#         latitude_degree = float(latitude_numerator)/float(latitude_denominator)
#         longitude = gps_info.get('GPSLongitude', None)
#         longitude_numerator, longitude_denominator = str((longitude)).split('/')
#         longitude_degree = float(longitude_numerator)/float(longitude_denominator)

#         fd = open(image, encoding='latin-1')
#         d = fd.read()
#         xmp_start = d.find('<x:xmpmeta')
#         xmp_end = d.find('</x:xmpmeta')
#         xmp_str = d[xmp_start:xmp_end]
#         yaw_start = xmp_str.find('drone-dji:FlightYawDegree="') + len('drone-dji:FlightYawDegree="')
#         yaw_end = xmp_str.find('"', yaw_start)
#         yaw = float(xmp_str[yaw_start:yaw_end])

#         print(f"Image size (width x height): {width} x {height}")
#         print(f"Latitude: {latitude_degree}")
#         print(f"Longitude: {longitude_degree}")
#         print(f"Altitude: {altitude}")
#         print(f'yaw: {yaw}')
#         x_center = image_locations[0][0]
#         y_center = image_locations[0][1]
#         print(f"center: {x_center}, {y_center}")

#         # locate.locate( latitude, longitude, altitude, bearing, cam_fov, imgWidth, imgHeight, obj_x_px, obj_y_px)
#         # cam_fov was consistently given as 73.7 in exif viewer
#         obj_long, obj_lat = locate.locate(latitude_degree, longitude_degree, altitude, yaw, 73.7, width, height, x_center, y_center )
#         # print(f"obj_latitude={obj_lat}, obj_longitude={obj_long}")
#         return [obj_long, obj_lat]

#     except Exception as e:
#         print(e)
#         return
#                                           # [long, lat]
# def AverageCoord(imageLocations: list[list[float, float]]):
#     totalLat = 0
#     totalLong = 0

#     for long, lat in imageLocations:
#         totalLong += long
#         totalLat += lat

#     averageLat = totalLat / len(imageLocations)
#     averageLong = totalLong / len(imageLocations)

#     print(f"obj_longitude={averageLong}, obj_latitude={averageLat}")

# if __name__ == "__main__":
#     # image = "/Users/dominicgartner/Desktop/SUAV dataset/pict/DJI_0679.JPG"
#     coord = []
#     images = ["/Users/dominicgartner/Desktop/SUAV/SUAV dataset/pict/DJI_0488.JPG", "/Users/dominicgartner/Desktop/SUAV/SUAV dataset/pict/DJI_0489.JPG", "/Users/dominicgartner/Desktop/SUAV/SUAV dataset/pict/DJI_0490.JPG", "/Users/dominicgartner/Desktop/SUAV/SUAV dataset/pict/DJI_0491.JPG"]
#     for image in images:
#         coord.append(getGPSPosition(image, [(0.5, 0.5)]))
    
#     AverageCoord(coord)



import locate
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def getGPSPosition(image_locations:list[tuple[float,float]], image: Image.Image) -> list[float]:
    try:
        img = Image.open(image)
        width, height = img.size
        exif_data = img._getexif()

        gps_info = {}
        if exif_data and 0x8825 in exif_data:
            # Extract GPSInfo from exif_data
            gps_info = {GPSTAGS.get(tag, tag): value for tag, value in exif_data[0x8825].items()}
            for key in ['GPSLatitude', 'GPSLongitude']:
                if key in gps_info:
                    degrees, minutes, seconds = gps_info[key]
                    gps_info[key] = degrees + (minutes / 60) + (seconds / 3600)

        # change the altitude to maryland california's altitude. Calgary (1045m), maryland (32m). Difference is 1045 - 32 = 1013
        altitude = float((gps_info.get('GPSAltitude', None)) - 1013)
        latitude = gps_info.get('GPSLatitude', None)
        latitude_numerator, latitude_denominator = str((latitude)).split('/')
        latitude_degree = float(latitude_numerator)/float(latitude_denominator)
        longitude = gps_info.get('GPSLongitude', None)
        longitude_numerator, longitude_denominator = str((longitude)).split('/')
        longitude_degree = float(longitude_numerator)/float(longitude_denominator)

        fd = open(image, encoding='latin-1')
        d = fd.read()
        xmp_start = d.find('<x:xmpmeta')
        xmp_end = d.find('</x:xmpmeta')
        xmp_str = d[xmp_start:xmp_end]
        yaw_start = xmp_str.find('drone-dji:FlightYawDegree="') + len('drone-dji:FlightYawDegree="')
        yaw_end = xmp_str.find('"', yaw_start)
        yaw = float(xmp_str[yaw_start:yaw_end])

        print(f"Image size (width x height): {width} x {height}")
        print(f"Latitude: {latitude_degree}")
        print(f"Longitude: {longitude_degree}")
        print(f"Altitude: {altitude}")
        print(f'yaw: {yaw}')
        x_center = image_locations[0][0]
        y_center = image_locations[0][1]
        print(f"center: {x_center}, {y_center}")

        # locate.locate( latitude, longitude, altitude, bearing, cam_fov, imgWidth, imgHeight, obj_x_px, obj_y_px)
        # cam_fov was consistently given as 73.7 in exif viewer
        obj_long, obj_lat = locate.locate(latitude_degree, longitude_degree, altitude, yaw, 73.7, width, height, x_center, y_center )
        # print(f"obj_latitude={obj_lat}, obj_longitude={obj_long}")
        return [obj_long, obj_lat]

    except Exception as e:
        print(e)
        return

if __name__ == "__main__":
    # image = "/Users/dominicgartner/Desktop/SUAV dataset/pict/DJI_0679.JPG"
    long, lat = getGPSPosition([(0.5, 0.5)], image="/Users/dominicgartner/Desktop/SUAV/SUAV dataset/pict/DJI_0488.JPG" )
    print(f"obj_latitude={lat}, obj_longitude={long}")
