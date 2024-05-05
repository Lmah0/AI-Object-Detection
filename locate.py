import math
from geographiclib.geodesic import Geodesic

# Parameters:
# uav_longitude - longitude of the center of the image in degrees
# uav_latitude - latitude of the center of the image in degrees
# uav_altitude - altitude of the uav in meters
# bearing - azimuth angle measured from North clockwise
# cam_fov - field of view of the camera in degrees
# img_width_px - width of the image in pixels
# img_height_px - height of the image in pixels
# obj_x_px - x coordinate of the target object in pixels where (0,0) is the center of the image
# obj_y_px - y coordinate of the target object in pixels
# Returns:
# (obj_longitude, obj_latitude) - the estimated longitude and latitude of the object

def locate(uav_latitude, uav_longitude, uav_altitude, bearing, cam_fov, img_width_px, img_height_px, obj_x_px, obj_y_px):
    cam_fov_rad = math.radians(cam_fov)
    diagonal_dist = math.tan(cam_fov_rad) * uav_altitude
    img_width = math.sqrt(diagonal_dist**2 / (1 + img_height_px / img_width_px)) # w^2 + (h/w)*w^2 = diag^2
    img_height = img_height_px / img_width_px * img_width
    length_per_px = img_width / img_width_px
    obj_x = obj_x_px * length_per_px
    obj_y = obj_y_px * length_per_px
    dist = math.sqrt(obj_x**2 + obj_y**2)

    angle = math.atan2(obj_y_px, obj_x_px) # Polar angle of the object point P
    true_bearing = (bearing + 90 - math.degrees(angle)) % 360
    geod = Geodesic.WGS84
    azil = true_bearing # Angle from North CW of the object point P
    g = geod.Direct(uav_latitude, uav_longitude, azil, dist)
    (obj_latitude, obj_longitude) = (g['lat2'], g['lon2'])
    return (obj_latitude, obj_longitude)


if __name__ == "__main__":
    uav_longitude = -128
    uav_latitude = 60
    uav_altitude = 1000
    bearing = 0
    cam_fov = 60
    img_width_px = 2000
    img_height_px = 1000
    obj_x_px = 1000
    obj_y_px = 500
    (obj_latitude, obj_longitude) = locate(uav_longitude=uav_longitude, uav_latitude=uav_latitude, uav_altitude=uav_altitude, bearing=bearing, cam_fov=cam_fov, img_width_px=img_width_px,
        img_height_px=img_height_px, obj_x_px=obj_x_px, obj_y_px=obj_y_px)
    print("obj_latitude={}, obj_longitude={}".format(obj_latitude, obj_longitude))