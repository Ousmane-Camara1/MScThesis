import cv2 

print("scanning for active cams...")

for index in range(10):
	cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
	if cap.isOpened():
		print("[success] camera found at index {index}")
		ret, frame = cap.read()
		if ret:
			print(f"   -> Frame captured. Resolution: {frame.shape[1]}x{frame.shape[0]}")
			print(f"   -> Use video_source = {index}")
		else:
			print(f"   -> opened but failed to read frame")
			
		cap.release()
	else:
		pass
print("Scan Complete")
