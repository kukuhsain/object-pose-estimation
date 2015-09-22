#import appropriate python modules to the program
import numpy as np
import cv2
from matplotlib import pyplot as plt
import freenect


# capturing video from Kinect Xbox 360
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

# callback function for selecting object by clicking 4-corner-points of the object
def select_object(event, x, y, flags, param):
    global box_pts, frame
    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(box_pts) < 4:
        box_pts.append([x, y])
        frame = cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)

# selecting object by clicking 4-corner-points
def select_object_mode():
    global input_mode, initialize_mode
    input_mode = True
    
    frame_static = frame.copy()

    while len(box_pts) < 4:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    
    initialize_mode = True
    input_mode = False

# setting the boundary of reference object
def set_boundary_of_reference(box_pts):
    
    ### upper bound ###
    if box_pts[0][1] < box_pts[1][1]:
        upper_bound = box_pts[0][1]
    else:
        upper_bound = box_pts[1][1]
    
    ### lower bound ###
    if box_pts[2][1] > box_pts[3][1]:
        lower_bound = box_pts[2][1]
    else:
        lower_bound = box_pts[3][1]
    
    ### left bound ###
    if box_pts[0][0] < box_pts[2][0]:
        left_bound = box_pts[0][0]
    else:
        left_bound = box_pts[2][0]
    
    ### right bound ###
    if box_pts[1][0] > box_pts[3][0]:
        right_bound = box_pts[1][0]
    else:
        right_bound = box_pts[3][0]
        
    upper_left_point = [0,0]
    upper_right_point = [(right_bound-left_bound),0]
    lower_left_point = [0,(lower_bound-upper_bound)]
    lower_right_point = [(right_bound-left_bound),(lower_bound-upper_bound)]
    
    pts2 = np.float32([upper_left_point, upper_right_point, lower_left_point, lower_right_point])
    
    # display dimension of reference object image to terminal
    print pts2
    
    return pts2, right_bound, left_bound, lower_bound, upper_bound

# doing perspective transform to reference object
def input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound):
    global object_orb
    pts1 = np.float32(box_pts)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_object = cv2.warpPerspective(frame,M,((right_bound-left_bound),(lower_bound-upper_bound)))
    return cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

# feature detection and description using ORB
def orb_feature_descriptor(img_object):
    kp1, des1 = orb.detectAndCompute(img_object,None)
    kp2, des2 = orb.detectAndCompute(frame,None)
    return kp1, des1, kp2, des2

# feature matching using Brute Force
def brute_force_feature_matcher(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key = lambda x:x.distance)

# finding homography matrix between reference and image frame
def find_homography_object(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M, mask

# applying homography matrix as inference of perpective transformation
def output_perspective_transform(img_object, M):
    h,w = img_object.shape
    corner_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
    corner_pts_3d = np.float32([ [-w/2,-h/2,0],[-w/2,(h-1)/2,0],[(w-1)/2,(h-1)/2,0],[(w-1)/2,-h/2,0] ])###
    corner_camera_coord = cv2.perspectiveTransform(corner_pts,M)###
    center_camera_coord = cv2.perspectiveTransform(center_pts,M)
    return corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts

# solving pnp using iterative LMA algorithm
def iterative_solve_pnp(object_points, image_points):
    image_points = image_points.reshape(-1,2)
    retval, rotation, translation = cv2.solvePnP(object_points, image_points, kinect_intrinsic_param, kinect_distortion_param)
    return rotation, translation

# drawing box around object
def draw_box_around_object(dst):
    return cv2.polylines(frame, [np.int32(dst)],True,255,3)
    
# recording sample data
def record_samples_data(translation, rotation):
    translation_list = translation.tolist()
    rotation_list = rotation.tolist()
    
    t1.append(translation_list[0])
    t2.append(translation_list[1])
    t3.append(translation_list[2])
    
    r1.append(rotation_list[0])
    r2.append(rotation_list[1])
    r3.append(rotation_list[2])
    
# computing and showing recorded data to terminal
def showing_recorded_data_to_terminal(t1, t2, t3, r1, r2, r3):
    
    # convert to numpy array
    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)
    
    # print mean and std of the data to terminal
    print "mean t1", np.mean(t1)
    print "std t1", np.std(t1)
    print ""
    print "mean t2", np.mean(t2)
    print "std t2", np.std(t2)
    print ""
    print "mean t3", np.mean(t3)
    print "std t3", np.std(t3)
    print ""
    print ""
    print "mean r1", np.mean(r1)
    print "std r1", np.std(r1)
    print ""
    print "mean r2", np.mean(r2)
    print "std r2", np.std(r2)
    print ""
    print "mean r3", np.mean(r3)
    print "std r3", np.std(r3)
    print ""
    print "#####################"
    print ""

# showing object position and orientation value to frame
def put_position_orientation_value_to_frame(translation, rotation):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,'position(cm)',(10,30), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(translation[0],2)),(250,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(translation[1],2)),(350,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(translation[2],2)),(450,30), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    
    cv2.putText(frame,'orientation(degree)',(10,60), font, 0.7,(0,255,0),1,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(rotation[0],2)),(250,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(rotation[1],2)),(350,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(rotation[2],2)),(450,60), font, 0.7,(0,0,255),2,cv2.LINE_AA)
    
    return frame


############
### Main ###
############

# initialization
input_mode = False
initialize_mode = False
track_mode = False
box_pts = []

record_num = 0
record_mode = False

t1, t2, t3, r1, r2, r3 = [], [], [], [], [], []

kinect_intrinsic_param = np.array([[514.04093664, 0., 320], [0., 514.87476583, 240], [0., 0., 1.]])
kinect_distortion_param = np.array([2.68661165e-01, -1.31720458e+00, -3.22098653e-03, -1.11578383e-03, 2.44470018e+00])

orb = cv2.ORB_create()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_object)

while True:
    
    frame = get_video()
    
    k = cv2.waitKey(1) & 0xFF
    
    # press i to enter input mode
    if k == ord('i'):
        
        # select object by clicking 4-corner-points
        select_object_mode()
        
        # set the boundary of reference object
        pts2, right_bound, left_bound, lower_bound, upper_bound = set_boundary_of_reference(box_pts)
        
        # do perspective transform to reference object
        img_object = input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound)
        
        track_mode = True
    
    # track mode is run immediately after user selects 4-corner-points of object
    if track_mode is True:
        # feature detection and description
        kp1, des1, kp2, des2 = orb_feature_descriptor(img_object)
        
        # feature matching
        matches = brute_force_feature_matcher(kp1, des1, kp2, des2)
        
        # find homography matrix
        M, mask = find_homography_object(kp1, kp2, matches)
        
        # apply homography matrix using perspective transformation
        corner_camera_coord, center_camera_coord, object_points_3d, center_pts = output_perspective_transform(img_object, M)
        
        # solve pnp using iterative LMA algorithm
        rotation, translation = iterative_solve_pnp(object_points_3d, corner_camera_coord)
        
        # convert to centimeters
        translation = (40./53.) * translation *.1
        
        # convert to degree
        rotation = rotation * 180./np.pi
        
        # press r to record 50 sample data and calculate its mean and std
        if k == ord("r"):
            record_mode = True
            
        if record_mode is True :
            record_num = record_num + 1
            
            # record 50 data
            record_samples_data(translation, rotation)
            
            if record_num == 50:
                record_mode = False
                record_num = 0
                
                # compute and show recorded data
                showing_recorded_data_to_terminal(t1, t2, t3, r1, r2, r3)
                
                # reset the data after 50 iterations
                t1, t2, t3, r1, r2, r3 = [], [], [], [], [], []
        
        # draw box around object
        frame = draw_box_around_object(corner_camera_coord)
        
        # show object position and orientation value to frame
        frame = put_position_orientation_value_to_frame(translation, rotation)
    
    cv2.imshow("frame", frame)
    
    # break when user pressing ESC
    if k == 27:
        break

cv2.destroyAllWindows()