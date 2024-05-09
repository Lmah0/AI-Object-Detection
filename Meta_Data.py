import locate
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def getGPSPosition(image: str, image_locations:tuple[float,float]) -> tuple[float|None, float|None]:
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

        # altitude = gps_info.get('GPSAltitude', None)
        latitude = gps_info.get('GPSLatitude', None)
        latitude_numerator, latitude_denominator = str((latitude)).split('/')
        latitude_degree = float(latitude_numerator)/float(latitude_denominator)
        longitude = gps_info.get('GPSLongitude', None)
        longitude_numerator, longitude_denominator = str((longitude)).split('/')
        # We are in the West so just add a negative to the longitude
        longitude_degree = -float(longitude_numerator)/float(longitude_denominator)

        fd = open(image, encoding='latin-1')
        d = fd.read()
        xmp_start = d.find('<x:xmpmeta')
        xmp_end = d.find('</x:xmpmeta')
        xmp_str = d[xmp_start+20:xmp_end+21]
        yaw_start = xmp_str.find('drone-dji:FlightYawDegree="') + len('drone-dji:FlightYawDegree="')
        yaw_end = xmp_str.find('"', yaw_start)
        yaw = float(xmp_str[yaw_start:yaw_end])

        alt_start = xmp_str.find('RelativeAltitude="') + len('RelativeAltitude="')
        alt_end = xmp_str.find('"', alt_start)
        alt = float(xmp_str[alt_start:alt_end])
        print(f"Image size (width x height): {width} x {height}")
        print(f"Latitude: {latitude_degree}")
        print(f"Longitude: {longitude_degree}")
        print(f"Altitude: {alt}")
        print(f'yaw: {yaw}')
        x_center = image_locations[0]
        y_center = image_locations[1]
        # convert img processing coordinates to calculation coordinates
        x_center = (x_center*2-1)*width/2
        y_center = -(y_center*2-1)*height/2

        print(f"center: {x_center}, {y_center}")

        # locate.locate( latitude, longitude, altitude, bearing, cam_fov, imgWidth, imgHeight, obj_x_px, obj_y_px)
        # cam_fov was consistently given as 73.7 in exif viewer
        obj_long, obj_lat = locate.locate(latitude_degree, longitude_degree, alt, yaw, 73.7, width, height, x_center, y_center )
        # print(f"obj_latitude={obj_lat}, obj_longitude={obj_long}")
        return obj_long, obj_lat
    except Exception as e:
        print(e)
        print("No exif")
        return 0, 0
if __name__ == "__main__":
    image = "D:/School/SUAV/dataset/Large/DJI_0491.JPG"
    gps_values = getGPSPosition(image, (1, 0.5))