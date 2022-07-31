# Human-Detection-through-CCTV


CCTV cameras as a piece of technology has improved the livelihood of several individuals, many testimonies justifying the need of these 
devices can be collected and put forth to advocate the need for technological advancements

The traditional surveillance system has not been able to solve the issue of safeguarding the restricted areas effectively. 
The cameras do not notify the owner about the crime or the unethical activity happening at that particular moment. 
Other concern is that cameras keep on recording each and every activity, in this process of recording each and every activity it becomes 
a very tedious process to find the particular video clip when the crime actually happened. To record and store every activity requires huge 
amount of storage space leading to the wastage of memory.

These problems would be removed by building a smart CCTV surveillance camera that would start recording only when humans enter the frame 
and automatically stops recording when there is no human in the frame. The owner can be notified the moment human enters the frame through 
alarm systems such as sending SMS and Emails. YOLO algorithm can be used to detect the presence of humans in the live video frame of CCTV and 
for capturing the images and videos OpenCV can be used.

Setup to be done as shown in figure.
![1](https://user-images.githubusercontent.com/63490088/182016625-619ff9b4-3b58-4524-832a-bf7c32d9b437.png)

![2](https://user-images.githubusercontent.com/63490088/182016676-57dc3664-929b-4d74-9b6f-536aa54dee69.png)


Downloading and Installation of Source code and other Dependencies

To download the weights file and the configuration file for the use of pre-trained YOLO
Algorithm. 
Visit the following link.

https://pjreddie.com/darknet/yolo/

This is optional as the weights and configuration files are already present in the source code.
For this project the weights and configuration file of YOLO version3 has been downloaded.
To install the complete source code in to local system, type the below command in terminal

git clone https://github.com/skgautam1010/Human-Detection-through-CCTV.git

Download Pip. To download pip the following command can be used
sudo apt update
sudo apt install python3-pip
pip install pipenv

Setup the virtual environment before downloading other dependencies.
Open terminal and go to the directory where project is located, then use the below command
pipenv shell
Download the other required libraries and packages
To download python3 use the following link
https://realpython.com/installing-python

To download all required packages and libraries, requirements.txt file has been created this
helps in downloading all the libraries with their versions with just a single command.
Run the following command in the virtual environment to download all the required
dependencies.

pip3 install -r requirements.txt

After downloading all the required packages migrate the complete project to Visual Studio
Code for better visibility of the entire project.
Activate the virtual environment with the command pipenv shell.
The project has the app.py file which is the main file to be executed.
Type the following command

python3 app.py

Go to the browser and type localhost:5000 to see the project up and running. The following
web page should appear which shows the project is executed successfully.

![3](https://user-images.githubusercontent.com/63490088/182016848-1caee7ac-a773-490b-be17-58b8c2c7ea7c.png)

![4](https://user-images.githubusercontent.com/63490088/182017127-1fbb8103-bc8c-4f82-bab4-1ea87530483c.png)
