import  cv2
import os
import cv2.aruco as aruco
import argparse
import glob
import numpy as np

from camera_calib import camera_calibration



def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--images_path", type = str, default = '')

	parser.add_argument("--do_calibration", dest="do_calibration", action="store_true")
	parser.add_argument("--done_calibration", dest="done_calibration", action="store_true")
	parser.add_argument("--create_markers", dest="create_markers", action="store_true")
	parser.add_argument("--run_camera", dest="run_camera", action="store_true")
	
	args = parser.parse_args()
	return args


def create_ArucoMarkers():

	print("Creating Markers -- 'DICT_4X4_50' ")
	aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) # Using 4x4 markers

	for i in range(50): # Please set the number 50 carefully... (Depends on Dict_4x4_***)

		img = aruco.drawMarker(aruco_dict, i, 500)
		cv2.imwrite('markers/4x4_ArucoMarkers'+str(i)+'.jpg',img)

	print("Saved markers into folder named 'markers' ")

	return aruco_dict


def detect_and_show_ArcuMarkers(image, aruco_dict, show=False):

	parameters = aruco.DetectorParameters_create()
	parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX 

	corners, ids, rejectedPoints = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

	if show:
		frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
		cv2.imshow('frame_markers',frame_markers)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return corners, ids, rejectedPoints


def get_world_coords_of_corners(file_path):

	file_ = open(file_path, "r")
	real_world_points = []
	num_ids = 0

	for line in file_:
		line_split = line.split(' ')

		if len(line_split) < 5:
			num_ids = int(line_split[0]) # Reading first line

		if len(line_split) == 5:
			ids_to_coords = []
			
			for i in range(len(line_split)):
				if i<2:
					ids_to_coords.append(int(line_split[i]))
				if i>=2:
					ids_to_coords.append(float(line_split[i]))

			real_world_points.append(ids_to_coords)	

	return num_ids, real_world_points


def track_markers_from_webcam(camera_matrix, dist_coef, aruco_dict):

	cap = cv2.VideoCapture(0)

	while(True):
		ret, frame = cap.read()
		if not ret:
			break


		parameters = aruco.DetectorParameters_create()
		parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

		corners, ids, rejectedPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
		rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.1776, camera_matrix, dist_coef)

		if ids is None:
			continue
		
		else:
			print(ids.size)
			for i in range(ids.size):

				aruco.drawAxis(frame, camera_matrix, dist_coef, rvec[i], tvec[i], 0.1)


			cv2.imshow('webcam', frame)
			if cv2.waitKey(50) & 0xFF == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()




if __name__ == '__main__':
	# -- This is where my main method will start...

	args = get_args()

	print(' ----  Starting the Main method...')

	# -- Can update with create_ArucoMarkers() and define new aruco dict in method 
	aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) # Default ArucoMarkers

	if args.create_markers:
		# Creating visual Aruco Markers with dict 4x4_50 ...
		aruco_dict = create_ArucoMarkers()

	if args.do_calibration:

		print('Starting the Camera Calibration.')
		
		camera_mat_txt = 'camera_matrix.txt'
		dist_coef_txt = 'distortion_coef.txt'
		camera_matrix, dist_coef = camera_calibration(camera_mat_txt, dist_coef_txt, False)
		
		print(' ----  Finished Camera Calibration.')

	if args.done_calibration:

		print('Loading the value of Camera_Matrix and Dist_Coef')

		cam_mat_txt  = open('camera_matrix.txt', 'r')
		camera_matrix = np.zeros((9,1),dtype=np.float32)
		cam_mat_elements = cam_mat_txt.read().split(' ')
		for i in range(9):
			camera_matrix[i] = cam_mat_elements[i]

		camera_matrix = camera_matrix.reshape(3,3)	
		
		dist_coef_txt = open('distortion_coef.txt', 'r')
		dist_coef = np.zeros((5,),dtype=np.float32)
		dist_coef_elements = dist_coef_txt.read().split(' ')
		for i in range(5):

			dist_coef[i] = dist_coef_elements[i]

		dist_coef = dist_coef.reshape(1,5)


	world_cords_f_path = 'aruco_corner_world_coord.txt'
	num_ids, real_world_points = get_world_coords_of_corners(world_cords_f_path)

	visited = [False for _ in range(num_ids)]

	object_points_in_world = []
	object_points_in_image = []

	image_num = 1

	file_3D_to_2D = open('Detected_ids_3D_2D.txt', 'w')

	for f_name in glob.glob(args.images_path+'/*.jpg'):

		print(f_name, " image number ---", image_num); image_num+=1

		image = cv2.imread(f_name, -1)
		image_copy = image.copy() # If we need at some point

		corners, ids, rejectedPoints = detect_and_show_ArcuMarkers(image, aruco_dict, False) # For showing parse last arg as True


		if(ids.shape[0] > 0): # Found arcuo marker
			

			for iter_ in range(ids.shape[0]):

				actual_id = ids[iter_]
				
				indexes = [j for j in range(len(real_world_points)) if real_world_points[j][0] == actual_id]

				for i in indexes:

					if not visited[i]:

						corner_num = real_world_points[i][1]

						X = real_world_points[i][2]
						Y = real_world_points[i][3]
						Z = real_world_points[i][4]

						world_point = np.zeros((1,3), dtype=np.float32)
						image_point = np.zeros((1,2), dtype=np.float32)

						world_point[0][0] = X
						world_point[0][1] = Y
						world_point[0][2] = Z

						# print(actual_id, " ",indexes  ," ", i , " ",corners[iter_][0][corner_num][0])

						image_point[0][0] = corners[iter_][0][corner_num][0]
						image_point[0][1] = corners[iter_][0][corner_num][1]
						
						file_3D_to_2D.write(str(int(actual_id)) + ' ' + str(world_point[0][0]) + " " + str(world_point[0][1]) + " " + str(world_point[0][2]) + " " + str(image_point[0][0]) + " " + str(image_point[0][1]) + '\n')

						object_points_in_world.append(world_point)
						object_points_in_image.append(image_point)


						visited[i] = True


	# Doing 3D correspondence using solvePnP					
	ret, rvec, tvec = cv2.solvePnP(np.array(object_points_in_world), np.array(object_points_in_image), camera_matrix, dist_coef)


	print("Rvec is -- ", rvec)
	print("Tvec is -- ", tvec)

	rotation_matrix = np.zeros(shape=(3,3))
	cv2.Rodrigues(rvec, rotation_matrix)

	translation_vector = np.zeros(shape=(3,1))
	translation_vector = tvec
	
	print("Rotation_Matrix in camera coordinates is -- ", rotation_matrix)
	print("Trans_Vector in camera coordinates is -- ", translation_vector)

	R_position = np.transpose(rotation_matrix)
	T_position = -1.0*np.transpose(rotation_matrix).dot(translation_vector)

	print("Rotation_Matrix in world coordinates is -- ", R_position)
	print("Trans_Vector in world coordinates is -- ", T_position)

	

	if args.run_camera:
		track_markers_from_webcam(camera_matrix, dist_coef, aruco_dict)

	print('Done!!!')






