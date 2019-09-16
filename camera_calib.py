import os
import cv2
import numpy as np
import glob

squareDimension = 0.01412 # In meters (chess square length measured by a ruler so might have errors)
sizeOfBoard = (6,9) # Size of the chess board

def get_chessboard_corners_in_world(sizeOfBoard, squareDimension):

	# This will create the world coordinates of corners (x,y,z=0 as its in a plane, and origin at top-left)
	world_corners = np.zeros((sizeOfBoard[1]*sizeOfBoard[0],3), np.float32)

	for j in range(sizeOfBoard[1]):
		for i in range(sizeOfBoard[0]):
			world_corners[i + j*sizeOfBoard[0],:] = np.array(([i*squareDimension, j*squareDimension, 0]), dtype=np.float64) 

	return world_corners


def get_chessboard_corners_in_image(images_path, sizeOfBoard, squareDimension, show=False):

	image_corners = []
	world_corners = []
	h = None
	w = None

	for f_name in glob.glob(images_path+'/*.jpg'):

		image = cv2.imread(f_name)

		if h is None and w is None: # Getting the image height and width needed for came
			h = image.shape[0]
			w = image.shape[1]

		found, corners = cv2.findChessboardCorners(image, sizeOfBoard)
		if found:
			image_corners.append(corners)
			world_corners.append(get_chessboard_corners_in_world(sizeOfBoard, squareDimension))

		if show:	
			# Showing the found corners
			drawn_image = image.copy()
			cv2.drawChessboardCorners(drawn_image, sizeOfBoard, corners, found)
			cv2.imshow('image detected -- ',drawn_image)
			cv2.waitKey(500)
			cv2.destroyAllWindows()

	return image_corners, world_corners, h, w			



def capture_images_from_camera(image_path, sizeOfBoard):
	
	cap = cv2.VideoCapture(0)

	image_num = 1
	while(True):
		ret, frame = cap.read()	

		if ret:
			found, corners = cv2.findChessboardCorners(frame, sizeOfBoard)
			
			if found:
				drawn_image = frame.copy()
				cv2.drawChessboardCorners(drawn_image, sizeOfBoard, corners, found)
				cv2.imshow('image -- ', drawn_image)
				cv2.waitKey(50)
				cv2.destroyAllWindows()

				cv2.imwrite(images_path +"/image_" + str(image_num) + ".jpg", frame)
				
				image_num+=1
			else:
				cv2.imshow('image -- ', frame)
				cv2.waitKey(50)	
				cv2.destroyAllWindows()

		if (image_num>=100): # will stop when stroed 100 images, you can change if need
			break			 		
	return



def camera_calibration(camera_mat, dist_coef, capture_images=False):

	if capture_images:
		image_path = 'calib_images'
		capture_images_from_camera(image_path, sizeOfBoard)

	else: # I have already selected 25 images so no need to run 'capture_images_from_camera()'
		images_path = '25_calib_images' 

	image_corners, world_corners, h, w = get_chessboard_corners_in_image(images_path, sizeOfBoard, squareDimension, False)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_corners, image_corners, (w, h), None, None)
	
	print("camera matrix:\n", mtx)
	print("distortion coefficients: ", dist)

	np.savetxt(camera_mat, mtx, newline=' ')
	np.savetxt(dist_coef, dist.ravel(), newline=' ')


	return mtx, dist