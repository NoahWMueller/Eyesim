import cv2
import numpy as np
import math
from Helper_Functions import load_map_points

# Load image
track_1 = cv2.imread('Sims/Worlds/Tracks/Track_1.png')
track_2 = cv2.imread('Sims/Worlds/Tracks/Track_2.png')

t1_img_width, t1_img_height = track_1.shape[1], track_1.shape[0]
t2_img_width, t2_img_height = track_2.shape[1], track_2.shape[0]

# Define the left lane coordinates
track_1_raw_left_lane_coordinates = [
    (106,42), (105,77), (151,42), (150,77),
    (192,42), (192,77), (227,42), (226,77),
    (268,42), (268,77), (305,43), (305,77),
    (349,42), (351,78), (392,43), (395,77),
    (426,43), (424,77), (463,57), (444,83),
    (488,75), (462,99), (505,106), (472,111),
    (513,129), (479,131), (513,170), (479,170),
    (513,207), (479,210), (513,248), (479,249),
    (513,284), (479,287), (513,324), (479,324),
    (513,355), (480,355), (506,389), (476,377),
    (493,418), (465,402), (467,447), (444,422),
    (439,466), (422,438), (402,479), (400,446),
    (355,479), (355,446), (315,480), (314,446),
    (272,481), (272,448), (235,482), (235,448),
    (190,481), (190,447), (145,481), (145,447),
    (102,481), (103,448), (50,458), (70,434),
    (21,420), (51,409), (13,380), (46,379),
    (22,341), (50,356), (44,309), (67,332),
    (70,284), (94,305), (92,268), (112,287),
    (116,243), (140,265), (139,219), (164,242),
    (164,196), (187,218), (183,177), (204,200),
    (212,146), (237,169), (265,129), (265,159),
    (321,142), (303,170), (358,183), (328,196),
    (369,217), (335,221), (370,260), (336,260),
    (370,301), (336,300), (359,333), (332,318),
    (342,362), (315,340), (302,388), (287,357),
    (253,396), (252,362), (200,376), (221,351),
    (171,348), (195,324), (146,324), (173,301),
    (136,313), (162,290), (112,287), (140,265),
    (92,265), (115,242), (64,237), (89,215),
    (32,203), (58,181), (13,153), (46,147),
    (25,97), (54,114), (54,63), (77,86),
    (106,42), (105,77)
]

track_1_raw_right_lane_coordinates = [
    (106, 110), (106, 77), (91, 115), (77, 87),
    (81, 127), (54, 114), (80, 145), (46, 147),
    (87, 165), (58, 181), (107, 183), (82, 207),
    (128, 207), (104, 231), (138, 218), (115, 242),
    (164, 242), (140, 265), (186, 268), (162, 290),
    (215, 299), (189, 318), (240, 323), (221, 351),
    (255, 328), (252, 362), (277, 325), (287, 357),
    (292, 315), (315, 340), (299, 305), (332, 318),
    (301, 300), (336, 300), (303, 263), (336, 260),
    (302, 222), (335, 221), (299, 214), (328, 196),
    (287, 202), (303, 170), (266, 196), (265, 159),
    (253, 198), (237, 169), (226, 229), (204, 200),
    (209, 242), (187, 218), (184, 266), (164, 242),
    (161, 287), (140, 265), (135, 315), (112, 287),
    (115, 332), (94, 305), (88, 358), (67, 332),
    (81, 369), (50, 356), (79, 381), (46, 379),
    (83, 397), (51, 409), (92, 406), (70, 434),
    (107, 414), (103, 448), (146, 415), (145, 447),
    (190, 415), (190, 447), (234, 415), (235, 448),
    (272, 415), (272, 448), (313, 414), (314, 446),
    (356, 414), (355, 446), (396, 413), (400, 446),
    (411, 407), (422, 438), (424, 396), (444, 422),
    (436, 382), (465, 402), (442, 366), (476, 377),
    (444, 354), (480, 355), (445, 325), (479, 324),
    (446, 287), (479, 287), (447, 250), (479, 249),
    (446, 212), (479, 210), (446, 172), (479, 170),
    (445, 137), (479, 131), (443, 127), (472, 111),
    (438, 122), (462, 99), (430, 116), (444, 83),
    (420, 111), (424, 77), (396, 108), (395, 77),
    (350, 108), (351, 78), (305, 108), (305, 77),
    (269, 109), (268, 77), (226, 110), (226, 77),
    (193, 110), (192, 77), (150, 109), (150, 77),
    (105, 111), (105, 77)
]

track_2_raw_left_lane_coordinates = load_map_points("Map_points/Track_2/raw_left_lane.txt")
track_2_raw_right_lane_coordinates = []

# Define the world dimensions
t1_world_width = 5000
t1_world_height = 5000

# Define the world dimensions
t2_world_width = 7500
t2_world_height = 7500

# Calculate scale factors
t1_scale_x = t1_world_width / t1_img_width
t1_scale_y = t1_world_height / t1_img_height

# Calculate scale factors
t2_scale_x = t2_world_width / t2_img_width
t2_scale_y = t2_world_height / t2_img_height

# Function to add polygons to the image
def image_processing(image,coordinates):
    centroids = []
    # Draw a small red circle at each coordinate and label with index
    for i, (x, y) in enumerate(coordinates[:-2]):
        # Draw the point on the image
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        # # Add a label with the point index: image, text, position, font, font scale, color, thickness
        cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # Creating polygons and finding centre point of each polygon
    for i in range(0, len(coordinates)-3, 2):
        # Create a polygon from the coordinates
        polygon = np.array([[coordinates[i],coordinates[i+1],coordinates[i+3],coordinates[i+2],coordinates[i]]], np.int32)
        
        # Draw the polygon on the image
        cv2.polylines(image, [polygon], isClosed=False, color=(0, 255, 0), thickness=2)

        # Calculate the centroid of the polygon
        centroid = np.mean(polygon, axis=1)
        x_centroid = int(centroid[0][0])
        y_centroid = int(centroid[0][1])
        
        # # Draw the centroid on the image
        # cv2.circle(image, (x_centroid, y_centroid), 5, (255, 0, 0), -1)

        # Find the midpoint of the two points of the polygon
        polygon = polygon[0]  # Extract the polygon points
        point1 = polygon[2]  # First point
        point2 = polygon[3]  # Second point
        midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

        # 3. Calculate the distance from the centroid to the midpoint
        dx = midpoint[0] - x_centroid
        dy = midpoint[1] - y_centroid

        # Use atan2 to get the angle (in radians)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)  # Convert radians to degrees

        # # Add a label with the point index: image, text, position, font, font scale, color, thickness
        cv2.putText(image, str(i//2), (x_centroid-5, y_centroid+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        centroids.append((x_centroid, y_centroid, round(angle_deg)))
    return image ,centroids


# Function to flip points based on the frame dimensions and save to file
def flip_points(points, img_width, img_height):
    flipped_points = []
    for x, y in points:
        new_x = img_width - x
        new_y = img_height - y
        flipped_points.append((new_x, new_y))

    return flipped_points

# Function to flip points based on the frame dimensions and save to file with angle
def flip_points_angles(points, img_width, img_height):
    flipped_points = []
    for x, y, phi in points:
        new_x = img_width - x
        new_y = img_height - y
        flipped_points.append((new_x, new_y, phi))

    return flipped_points

# Function to convert points to world coordinates and save to file
def convert_to_world_coordinates(points, filename, scale_x, scale_y, img_height):
    world_coordinates = [
        (round(px * scale_x), round((img_height - py) * scale_y))
        for (px, py) in points
    ]
    with open(f"{filename}.txt", "w") as file:
        for x, y in world_coordinates:
            file.write(f"({x},{y})\n")

# Function to convert points to world coordinates and save to file with angle
def convert_to_world_coordinates_angle(points, filename, scale_x, scale_y, img_height):
    world_coordinates = [
        (round(px * scale_x), round((img_height - py) * scale_y), phi)
        for (px, py , phi) in points
    ]
    with open(f"{filename}.txt", "w") as file:
        for x, y, phi in world_coordinates:
            file.write(f"({x},{y},{phi})\n")

# Complete conversion from image to world coordinates
def conversion(image, filename, track, left_lane_coordinates, right_lane_coordinates, img_width, img_height, scale_x, scale_y):
    image, left_centroids = image_processing(image,left_lane_coordinates)
    image, right_centroids = image_processing(image,right_lane_coordinates)
    cv2.imshow('Image', image)
    cv2.imwrite(filename, image)

    # Flip points and save to file
    flipped_left_lanepoints = flip_points(left_lane_coordinates, img_width, img_height)
    flipped_left_centroids = flip_points_angles(left_centroids, img_width, img_height)
    convert_to_world_coordinates(flipped_left_lanepoints, f"Map_points/Track_{track}/left_lane", scale_x, scale_y, img_height)
    convert_to_world_coordinates_angle(flipped_left_centroids, f"Map_points/Track_{track}/left_centroids", scale_x, scale_y, img_height)

    
    flipped_right_lanepoints = flip_points(right_lane_coordinates, img_width, img_height)
    flipped_right_centroids = flip_points_angles(right_centroids, img_width, img_height)
    convert_to_world_coordinates(flipped_right_lanepoints, f"Map_points/Track_{track}/right_lane", scale_x, scale_y, img_height)
    convert_to_world_coordinates_angle(flipped_right_centroids, f"Map_points/Track_{track}/right_centroids", scale_x, scale_y, img_height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
conversion(track_1, 'images-videos/Track_1_Centroids.png', 1, track_1_raw_left_lane_coordinates, track_1_raw_right_lane_coordinates, t1_img_width, t1_img_height, t1_scale_x, t1_scale_y)
conversion(track_2, 'images-videos/Track_2_Centroids.png', 2, track_2_raw_left_lane_coordinates, track_2_raw_right_lane_coordinates, t2_img_width, t2_img_height, t2_scale_x, t2_scale_y)