import numpy as np
from roboflow import Roboflow

### ROBOFLOW API ### 
rf = Roboflow(api_key="u3aLSMNORqVafnzaMShL")
project = rf.workspace().project("person-detection-9a6mk")
model = project.version(16).model

### PARAMETERS TO SET ###

queue_image1 = "/path/to/queue_image1.jpg"
queue_image2 = "/path/to/queue_image2.jpg"
time_difference = 1 # in minutes
polynomial_degree = 5
threshold = 3.9

# infer on a local image
json_file1 = (model.predict(queue_image1, confidence=50, overlap=50).json())
json_file2 = (model.predict(queue_image2, confidence=50, overlap=50).json())

# visualize prediction
# model.predict(queue_image1, confidence=50, overlap=50).save("prediction1.jpg")
# model.predict(queue_image1, confidence=50, overlap=50).save("prediction2.jpg")

x_coords1 = []
y_coords1 = []
for pred in json_file1["predictions"]:
    x = pred["x"]
    y = pred["y"]
    height = pred["height"]
    head_y = y - (height*0.4)
    x_coords1.append(x)
    y_coords1.append(head_y)

x_coords1 = np.array(x_coords1)
y_coords1 = np.array(y_coords1)

x_coords2 = []
y_coords2 = []
for pred in json_file2["predictions"]:
    x = pred["x"]
    y = pred["y"]
    height = pred["height"]
    head_y = y - (height*0.4)
    x_coords2.append(x)
    y_coords2.append(head_y)

x_coords2 = np.array(x_coords2)
y_coords2 = np.array(y_coords2)

### INLIER FUNCTIONS ###
def index_closest(lst, x):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - x))

def polynomial_regression(x, y, degree, threshold):
    """
    Calculate and plot the polynomial regression for a set of points, along with threshold lines.

    Parameters:
    x (list or np.array): x-coordinates of the points
    y (list or np.array): y-coordinates of the points
    degree (int): Degree of the polynomial
    threshold (float): The threshold distance from the regression line
    """
    
    count=0
    x = np.array(x)
    y = np.array(y)

    coefficients = np.polyfit(x, y, degree)

    # Generate a range of x values for plotting the polynomial line and threshold lines
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = np.polyval(coefficients, x_fit)

    # threshold lines
    y_fit_upper = y_fit + threshold
    y_fit_lower = y_fit - threshold
    
    for xx,yy in zip(x,y):
        x1=index_closest(x_fit,xx)
        y1_up=y_fit_upper[x1]
        y1_low=y_fit_lower[x1]
        
        if y1_low<yy<y1_up:
            count+=1
    
    return count

### GET COUNT ###
count1 = polynomial_regression(x_coords1, y_coords1,threshold, polynomial_degree)
count2 = polynomial_regression(x_coords2, y_coords2,threshold, polynomial_degree)

### GET DIFFERENCE ###
count_difference = count2 - count1

if count_difference < 0:
    print("Error: there are more people in the second picture than the first. Make sure the order is correct and that you have waited 5 mins before taking another picture.")

time_prediction = count2 * (count_difference/time_difference)
