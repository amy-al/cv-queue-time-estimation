# Queue Waiting Time Estimation using AI and CV techniques
AI-supported queue waiting-time estimation using crowd-counting in images supported by other computer vision techniques. 
Implemented using an R-CNN trained on the ShanghaiTech crowd dataset.

# How it works
1. Input two images at two different points in time (images are from perspective of a person standing in queue)
2. Input the images into the trained R-CNN model to get the point detections of people
3. Determine outliers, the people who are out of the queue to determine the total count of people in line (inspired by computer vision algorithm RANSAC (RANdom SAmple Consensus))
4. Calculate the rate of change in people/time to determine the estimated remaining waiting time.

# Contributors
Amy Li, Nayel Benabdesadok, Marc Bruni, Sidney Gharib

# References
This project builds off of the R-CNN model provided by Mehreen Tahir. 
The source code can be found here: https://www.codeproject.com/Articles/5283660/AI-Queue-Length-Detection-Counting-the-Number-of-P
