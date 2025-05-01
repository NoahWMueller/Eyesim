import cv2
import numpy as np
import random
import math

# Load image
image = cv2.imread('Sims/Carolo.png')


# (70,284),(94,305),(92,268),(112,287),(116,243),(140,265),(139,219),(164,242),(164,196),(187,218)
# 35 - 36 

# (190,320),(170,346),(164,294),(138,313),(138,265),(114,286),(114,242),(92,263),(88,211),(63,235)
# 59 - 60

# Define the left lane coordinates
left_lane_coordinates = [
    (106,42),(105,77),(426,43),(424,77),(463,57), 
    (444,83),(488,75),(462,99),(505,106),(472,111),
    (513,129),(479,131),(513,355),(480,355),(506,389),
    (476,377),(493,418),(465,402),(467,447),(444,422),
    (439,466),(422,438),(402,479),(400,446),(102,481),
    (103,448),(50,458),(70,434),(21,420),(51,409),
    (13,380),(46,379),(22,341),(50,356),(44,309),
    (67,332),(70,284),(94,305),(92,268),(112,287),
    (116,243),(140,265),(139,219),(164,242),(164,196),
    (187,218),(183,177),(204,200),(212,146),(237,169),
    (265,129),(265,159),(321,142),(303,170),(358,183),
    (328,196),(369,217),(335,221),(370,301),(336,300),
    (359,333),(332,318),(342,362),(315,340),(302,388),
    (287,357),(253,396),(252,362),(200,376),(221,351),
    (171,348),(195,324),(136,313),(162,290),(92,265),
    (115,242),(64,237),(89,215),(32,203),(58,181),
    (13,153), (46,147),(25,97),(54,114),(54,63),
    (77,86),(106,42),(105,77)
]

# Define the flipped left lane coordinates
flipped_left_lane_coordinates = [
    (419,483),(420,448),(99,482),(101,448),(62,468),
    (81,442),(37,450),(63,426),(20,419),(53,414),
    (12,396),(46,394),(12,170),(45,170),(19,136),
    (49,148),(32,107),(60,123),(58,78),(81,103),
    (86,59),(103,87),(123,46),(125,79),(423,44),
    (422,77),(475,67),(455,91),(504,105),(474,116),
    (512,145),(479,146),(503,184),(475,169),(481,216),
    (458,193),(455,241),(431,220),(433,257),(413,238),
    (409,282),(385,260),(386,306),(361,283),(361,329),
    (338,307),(342,348),(321,325),(313,379),(288,356),
    (260,396),(260,366),(204,383),(222,355),(167,342),
    (197,329),(156,308),(190,304),(155,224),(189,225),
    (166,192),(193,207),(183,163),(210,185),(223,137),
    (238,168),(272,129),(273,163),(325,149),(304,174),
    (354,177),(330,201),(389,212),(363,235),(433,260),
    (410,283),(461,288),(436,310),(493,322),(467,344),
    (512,372),(479,378),(500,428),(471,411),(471,462),
    (448,439),(419,483),(420,448)
]



# Define the image dimensions (from Carolo.png)
img_width = 525
img_height = 525

# Define the world dimensions
world_width = 5000
world_height = 5000

# Calculate scale factors
scale_x = world_width / img_width
scale_y = world_height / img_height

# Function to add polygons to the image
def image_processing(image, file_write=True):
    with open("points/centroids.txt", "w") as file:
        # Draw a small red circle at each coordinate and label with index
        for i, (x, y) in enumerate(left_lane_coordinates[:-2]):
            print(i , x, y)
            # Draw the point on the image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            # Add a label with the point index: image, text, position, font, font scale, color, thickness
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # Creating polygons and finding centre point of each polygon
        for i in range(0, len(left_lane_coordinates)-2, 2):
            # Create a polygon from the coordinates
            polygon = np.array([[left_lane_coordinates[i],left_lane_coordinates[i+1],left_lane_coordinates[i+3],left_lane_coordinates[i+2],left_lane_coordinates[i]]], np.int32)
            
            # Draw the polygon on the image
            cv2.polylines(image, [polygon], isClosed=False, color=(0, 255, 0), thickness=2)

            # Calculate the centroid of the polygon
            centroid = np.mean(polygon, axis=1)
            x_centroid = int(centroid[0][0])
            y_centroid = int(centroid[0][1])
            
            # Draw the centroid on the image
            cv2.circle(image, (x_centroid, y_centroid), 5, (255, 0, 0), -1)

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

            # Draw an arrow to indicate the direction
            cv2.arrowedLine(image, (int(x_centroid), int(y_centroid)), (int(midpoint[0]), int(midpoint[1])), (255, 255, 255), 2)

            # Add a label with the point index: image, text, position, font, font scale, color, thickness
            cv2.putText(image, str(i//2), (x_centroid, y_centroid-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if file_write == True :
                file.write(f"({x_centroid},{y_centroid},{round(angle_deg)}),")
                if (i//2) % 5 == 0 and i > 0: 
                    file.write("\n")
    return image

# Function to check if a point is inside any of the polygons
def point_checker(point):
    result = 0
    for i in range(0, len(left_lane_coordinates)-2, 2):
        polygon = np.array([
            left_lane_coordinates[i],
            left_lane_coordinates[i+1],
            left_lane_coordinates[i+3],
            left_lane_coordinates[i+2]
        ], np.int32)

        # Reshape the polygon points
        polygon = polygon.reshape((-1, 1, 2))

        # Check if the point is inside the polygon
        result = cv2.pointPolygonTest(polygon, point, False)

        # If the point is inside the polygon return
        if result > 0:
            print(f"Point {point} is inside the polygon {i}")
            break

# Function to flip points based on the frame dimensions and save to file
def flip_points(points, filename):
    flipped_points = []
    for x, y in points:
        new_x = img_width - x
        new_y = img_height - y
        flipped_points.append((new_x, new_y))

    with open(f"{filename}.txt", "w") as file:
        i = 1
        for x, y in flipped_points:
            file.write(f"({x},{y}),")
            if i % 5 == 0: file.write("\n")
            i+=1

# Function to flip points based on the frame dimensions and save to file with angle
def flip_points_angles(points, filename):
    flipped_points = []
    for x, y, phi in points:
        new_x = img_width - x
        new_y = img_height - y
        flipped_points.append((new_x, new_y, phi))

    with open(f"{filename}.txt", "w") as file:
        i = 1
        for x, y, phi in flipped_points:
            file.write(f"({x},{y},{phi}),")
            if i % 5 == 0: file.write("\n")
            i+=1


# Function to convert points to world coordinates and save to file
def convert_to_world_coordinates(points, filename):
    world_coordinates = [
        (round(px * scale_x), round((img_height - py) * scale_y))
        for (px, py) in points
    ]
    with open(f"{filename}.txt", "w") as file:
        i = 1
        for x, y in world_coordinates:
            file.write(f"({x},{y}),")
            if i % 5 == 0: file.write("\n")
            i+=1

# Function to convert points to world coordinates and save to file with angle
def convert_to_world_coordinates_angle(points, filename):
    world_coordinates = [
        (round(px * scale_x), round((img_height - py) * scale_y), phi)
        for (px, py , phi) in points
    ]
    with open(f"{filename}.txt", "w") as file:
        i = 1
        for x, y, phi in world_coordinates:
            file.write(f"({x},{y},{phi}),")
            if i % 5 == 0: file.write("\n")
            i+=1

# Function to save the image
def save_image(image, filename):
    image = image_processing(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(filename, image)

convert_to_world_coordinates(flipped_left_lane_coordinates, "points/flipped_left_lane_world_coordinates")

# convert_to_world_coordinates_angle(coordinates, "points/flipped_centroids_world_coordinates")
save_image(image, 'Carolo_labelled.png')
