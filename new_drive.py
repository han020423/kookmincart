#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=============================================
# 본 프로그램은 2025 제8회 국민대 자율주행 경진대회에서
# 예선과제를 수행하기 위한 파일입니다.
# 예선과제 수행 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터 제어 ROS Publisher
motor_msg = XycarMotor()  # 모터 토픽 메시지
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지

# 주행 상태 관련 변수
current_mission_state = "WAITING_FOR_GREEN_LIGHT"
# 차량 제어 상수
Fix_Speed = 60  # 기본 주행 속도
Slow_Speed = 20
Min_Speed = 5   # 최소 주행 속도 (예: 장애물 회피 시)
Max_Speed = 100  # 최대 주행 속도
Angle_Limit = 75 # 최대 조향각
prev_angle = 0.0 #이전 조향각
last_avoidance_time = 0  # 마지막 차선변경 시각 저장용 (초)
AVOIDANCE_COOLDOWN = 8   # 쿨타임 (초)

# 이미지 처리 관련 변수
TRAFFIC_LIGHT_ROI_INITIAL = (280, 100, 80, 150) 
# HSV 색상 범위 (Green, Yellow, Red)
GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([90, 255, 255])
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])
RED_LOWER1 = np.array([0, 100, 100]) 
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# 차선 감지 ROI 
LANE_ROI_VERT_PERCENT_START = 0.6 
LANE_ROI_VERT_PERCENT_END = 0.95   

#=============================================
# 라이다 스캔정보로 그림을 그리기 위한 변수 (디버깅용)
#=============================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 10) #LIDAR 최대 감지 거리가 100m 이나, 실제 주행에 중요한 영역 위주로 표시
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo', markersize=2)

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지를 받아 image 변수에 저장
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8") # ROS Image 메시지를 OpenCV 이미지로 변환

#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
# 라이다 데이터를 받아 ranges 변수에 저장
#=============================================
def lidar_callback(data):
    global ranges
    ranges = np.array(data.ranges)

#=============================================
# 모터로 토픽을 발행하는 함수
# 주어진 각도(angle)와 속도(speed)로 차량을 제어
#=============================================
def drive(angle, speed):
    motor_msg.angle = max(min(float(angle), Angle_Limit), -Angle_Limit)
    motor_msg.speed = max(min(float(speed), Max_Speed), -Max_Speed)
    motor.publish(motor_msg)

#=============================================
# 신호등 인식 함수
# 입력: 현재 카메라 이미지 (BGR)
# 출력: "RED", "YELLOW", "GREEN", 또는 "NONE"
#=============================================
def detect_traffic_light(current_image):
    if current_image.size == 0:
        return "NONE"

    height, width = current_image.shape[:2]
    x, y, w, h = int(width*0.2), int(height*0.05), int(width*0.6), int(height*0.2)

    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return "NONE" 

    roi = current_image[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3. 색상 마스크 생성 및 빛 감지
    # 각 색상(빨강, 노랑, 초록)에 대한 마스크 생성
    mask_green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 4. 가장 두드러지는 색상 판단
    # 각 마스크에서 0이 아닌 픽셀 수를 계산하여 어떤 색이 가장 많이 검출되었는지 확인
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    red_pixels = cv2.countNonZero(mask_red)
    
    # cv2.imshow("Traffic Light ROI", roi) # 디버깅용
    # cv2.imshow("Green Mask", mask_green) # 디버깅용

    # 임계값 설정 
    min_pixel_threshold = 50 
    if green_pixels > min_pixel_threshold and green_pixels > yellow_pixels and green_pixels > red_pixels:
        # print("Traffic Light: GREEN")
        return "GREEN"
    elif yellow_pixels > min_pixel_threshold and yellow_pixels > green_pixels and yellow_pixels > red_pixels:
        # print("Traffic Light: YELLOW")
        return "YELLOW"
    elif red_pixels > min_pixel_threshold and red_pixels > green_pixels and red_pixels > yellow_pixels:
        # print("Traffic Light: RED")
        return "RED"
    
    return "NONE" 


#=============================================
# 차량 위치 판단
#=============================================
def detect_yellow_lane_side(image, left_lines, right_lines):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_range = (15, 50)  # Hue 기준

    def yellow_ratio_along_lines(lines):
        yellow_count = 0
        total_count = 0

        for x1, y1, x2, y2 in lines:
            num_points = 10
            for i in range(num_points):
                x = int(x1 + (x2 - x1) * i / (num_points - 1))
                y = int(y1 + (y2 - y1) * i / (num_points - 1))
                if 0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]:
                    hue = hsv[y, x, 0]
                    if yellow_range[0] <= hue <= yellow_range[1]:
                        yellow_count += 1
                    total_count += 1

        if total_count == 0:
            return 0.0
        return yellow_count / total_count

    left_yellow_ratio = yellow_ratio_along_lines(left_lines)
    right_yellow_ratio = yellow_ratio_along_lines(right_lines)

    # print(f"[Hue 분석] 왼쪽 노란색 비율: {left_yellow_ratio:.2f}, 오른쪽: {right_yellow_ratio:.2f}")

    if left_yellow_ratio > right_yellow_ratio and left_yellow_ratio > 0.1:
        return "Right Lane"  # 중앙선이 왼쪽에 있음 → 우측 차선 주행 중
    elif right_yellow_ratio > left_yellow_ratio and right_yellow_ratio > 0.1:
        return "Left Lane"   # 중앙선이 오른쪽에 있음 → 좌측 차선 주행 중
    else:
        return "Unknown"


#=============================================
# 차선 인식 및 조향각 계산 함수
# 입력: 현재 카메라 이미지
# 출력: 계산된 조향각 
#=============================================
def calculate_lane_steering(current_image):
    if current_image.size == 0:
        return 0.0

    # 이미지 전처리 
    height, width = current_image.shape[:2]
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ROI 설정: 화면 하단 영역만 사용
    roi_start_y = int(height * LANE_ROI_VERT_PERCENT_START)
    roi_end_y = int(height * LANE_ROI_VERT_PERCENT_END)

    roi_mask = np.zeros_like(blur)
    poly_vertices = np.array([[(0, roi_end_y), (width*0.1, roi_start_y), (width*0.9, roi_start_y), (width, roi_end_y)]], dtype=np.int32)
    cv2.fillPoly(roi_mask, poly_vertices, 255)
    
    masked_image = cv2.bitwise_and(blur, roi_mask)
    #cv2.imshow("Lane ROI", masked_image) # 디버깅용

    edges = cv2.Canny(masked_image, 70, 140)
    # cv2.imshow("Lane Edges", edges) # 디버깅용

    # 허프 변환 선분 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=20) 

    if lines is None:
        return 0.0 # 차선 미검출 시 직진

    # 검출된 선들을 이용해 좌/우 차선 대표선 계산 및 화면 중앙과의 오차 계산
    # 화면 중앙과 차선 중앙 간의 차이 계산해  비례하는 조향각을 설정
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        if slope < -0.3: # 좌측 차선 
            left_lines.append(line[0])
        elif slope > 0.3: # 우측 차선 
            right_lines.append(line[0])


    # 대표선 계산
    # 좌측 차선이 인식된 경우
    if left_lines:
        left_x_coords = []
        for x1,y1,x2,y2 in left_lines:
            left_x_coords.extend([x1,x2])
        avg_left_x = np.mean(left_x_coords)
    else: 
        avg_left_x = width * 0.1 

    # 우측 차선이 인식된 경우
    if right_lines:
        right_x_coords = []
        for x1,y1,x2,y2 in right_lines:
            right_x_coords.extend([x1,x2])
        avg_right_x = np.mean(right_x_coords)
    else: 
        avg_right_x = width * 0.9

    lane_center_x = (avg_left_x + avg_right_x) / 2.0
    car_center_x = width / 2.0
    
    error = lane_center_x - car_center_x

    # 조향각 계산 
    Kp = 0.4 
    steering_angle = Kp * error
    

    debug_image = current_image.copy()
    # 좌측 차선 선분은 파란색
    for x1, y1, x2, y2 in left_lines:
        cv2.line(current_image, (x1, y1), (x2, y2), (255, 0, 0), 2) 
    # 우측 차선 선분은 빨간색색
    for x1, y1, x2, y2 in right_lines:
        cv2.line(current_image, (x1, y1), (x2, y2), (0, 0, 255), 2) 

    # ROI 선 시각화
    cv2.polylines(current_image, [poly_vertices], isClosed=True, color=(0, 255, 0), thickness=2)


    # 색상 분석 
    lane_position = detect_yellow_lane_side(debug_image, left_lines, right_lines)
 
    return steering_angle, lane_position


def visualize_yellow_area(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    yellow_mask = cv2.inRange(hue, 15, 50)

    yellow_highlight = image_bgr.copy()
    yellow_highlight[yellow_mask > 0] = [0, 0, 255]
    cv2.imshow("Yellow Lane Highlight", yellow_highlight)
    cv2.waitKey(1)


#=============================================
# 라바콘 감지
#=============================================
def detect_cone_presence(current_image, lidar_data):
    if lidar_data is None or current_image.size == 0:
        return False

    # 1. LiDAR 기준 유효한 거리값 추출
    front_left = lidar_data[340:]
    front_right = lidar_data[:21]
    front_scan = np.concatenate((front_left, front_right))
    front_scan = np.array(front_scan)
    valid_lidar = front_scan[np.isfinite(front_scan) & (front_scan > 0.5) & (front_scan < 10.0)]

    # 2. 5개 이상 감지된 경우에만 카메라 판단 수행
    if len(valid_lidar) < 4:
        return False

    height = current_image.shape[0]
    roi = current_image[int(height * 0.6):, :]

    # 3. 카메라에서 라바콘 색상(Hue ≈ 8) 필터링
    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
    LOWER_CONE = np.array([5, 80, 150])
    UPPER_CONE = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, LOWER_CONE, UPPER_CONE)

    # 4. 엣지(윤곽선) 검출
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 일정 크기 이상의 윤곽선이 존재하면 라바콘으로 판단
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  
            return True

    return False

def detect_cones_only_camera(image):
    height, width = image.shape[:2]
    roi = image[int(height * 0.6):, :]  # 하단 40% 사용
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    LOWER_CONE = np.array([5, 80, 150])
    UPPER_CONE = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, LOWER_CONE, UPPER_CONE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 일정 크기 이상의 라바콘이 존재하는지 확인
    return any(cv2.contourArea(cnt) > 100 for cnt in contours)


#=============================================
# 라바콘 주행
#=============================================
def calculate_cone_steering_camera(image, prev_angle=0, debug=False):

    if image is None or image.size == 0:
        return prev_angle

    height, width = image.shape[:2]
    roi_start = int(height * 0.7)
    roi = image[roi_start:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    LOWER_ORANGE = np.array([5, 100, 100])
    UPPER_ORANGE = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    # 중앙 무시 영역 정의 (45%~55%)
    center_ignore_min = int(width * 0.5)
    center_ignore_max = int(width * 0.5)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # 중앙 영역 무시
                if center_ignore_min <= cx <= center_ignore_max:
                    continue
                centers.append((cx, cy))

    vis = roi.copy()
    left = []
    right = []
    for cx, cy in centers:
        if cx < width // 2:
            left.append(cx)
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
        else:
            right.append(cx)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)


    angle = prev_angle

    if len(left) > 0 and len(right) > 0:
        left_x = min(left)
        right_x = max(right)
        gap = right_x - left_x

        if gap < 350:
            if debug: print(f"[CAMERA] 라바콘 간격 {gap}px < 350 → 이전 각도 유지")
        else:
            mid_x = (left_x + right_x) // 2
            error = mid_x - width // 2
            angle = np.clip(0.6 * error, -40, 40)
            if debug:
                print(f"[CAMERA] 중앙: {mid_x}, 간격: {gap}, 조향: {angle:.2f}")
                cv2.line(vis, (mid_x, 0), (mid_x, vis.shape[0]), (0, 255, 0), 2)
                cv2.line(vis, (width // 2, 0), (width // 2, vis.shape[0]), (0, 255, 255), 2)

    elif len(left) > 0:
        left_x = min(left)
        angle = 75
        if debug: print("[CAMERA] 왼쪽 라바콘만 감지 → 우회전")

    elif len(right) > 0:
        right_x = max(right)
        angle = -75
        if debug: print("[CAMERA] 오른쪽 라바콘만 감지 → 좌회전")

    else:
        if debug: print("[CAMERA] 라바콘 없음 → 이전 각도 유지")

    if debug:
        cv2.imshow("Cone Detection (camera)", vis)
        cv2.rectangle(vis, (center_ignore_min, 0), (center_ignore_max, vis.shape[0]), (0, 255, 255), 2)
        cv2.waitKey(1)

    return angle



#=============================================
# 차량 판단 함수
#=============================================
def detect_vehicle_front(lidar_data):
    if lidar_data is None or len(lidar_data) < 360:
        return "CLEAR"

    front_left = lidar_data[355:]
    front_right = lidar_data[:6]
    front_scan = np.concatenate((front_left, front_right))

    front_scan = np.array(front_scan)
    valid = front_scan[np.isfinite(front_scan) & (front_scan > 1.0) & (front_scan < 20.0)]

    if len(valid) == 0:
        return "CLEAR"

    # 연속 감지 구간 계산
    close_flags = (front_scan > 1.0) & (front_scan < 20.0)
    consecutive_count = 0
    max_consecutive = 0

    for val in close_flags:
        if val:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
        else:
            consecutive_count = 0

    print(f"[전방 감지] 유효 거리 {len(valid)}개 / 연속 {max_consecutive}개")

    # 🚗 차량으로 판단 (완전 정지)
    if max_consecutive >= 8:
        return "STOP"

    # 🐢 감지는 됐지만 확신 부족 (감속)
    elif len(valid) >= 5:
        return "SLOW"

    return "CLEAR"

#=============================================
# 메인 함수
#=============================================
def start():
    global motor, image, ranges, current_mission_state, prev_angle, last_avoidance_time
    lane_position = "Unknown"
    print("Start program --------------")

    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
        
    #=========================================
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 대기
    #=========================================
    print("Waiting for first messages...")
    rospy.wait_for_message("/usb_cam/image_raw/", Image) 
    print("Camera Ready --------------")
    rospy.wait_for_message("/scan", LaserScan) 
    print("Lidar Ready --------------")
    
    #=========================================
    # 라이다 스캔정보 시각화 준비
    #=========================================
    if 'matplotlib' in sys.modules: 
        plt.ion()
        plt.show()
        print("Lidar Visualizer Ready (matplotlib) ----------")
    else:
        print("Matplotlib not available for Lidar visualization.")

    
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")

    #=========================================
    # 메인 루프 
    # 루프 주기 설정
    #=========================================
    loop_rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # 현재 이미지와 라이다 데이터 복사
        current_image_data = image.copy()
        current_lidar_data = ranges.copy() if ranges is not None else None

        # 조향각과 속도 초기화
        target_angle = 0.0
        target_speed = 0.0

        # ========================================
        # 주행 상태에 따른 로직 분기
        # ========================================
        if current_mission_state == "WAITING_FOR_GREEN_LIGHT":
            # 신호등이 녹색이 될 때까지 대기
            traffic_light_color = detect_traffic_light(current_image_data)
            # print(f"STATE: WAITING_FOR_GREEN_LIGHT, Detected: {traffic_light_color}")
            if traffic_light_color == "GREEN":
                print("Green light detected! Starting lane following.")
                current_mission_state = "LANE_FOLLOWING"
                target_speed = Fix_Speed # 출발 속도
            elif traffic_light_color == "YELLOW" or traffic_light_color == "RED":
                target_speed = 0 
            else: 
                target_speed = 0 

        elif current_mission_state == "LANE_FOLLOWING":
            # 차선 주행 로직
            # print("STATE: LANE_FOLLOWING")
            target_angle, lane_position = calculate_lane_steering(current_image_data)
            target_speed = Fix_Speed

            # 라바콘 감지 시 상태 전환
            if detect_cone_presence(current_image_data, current_lidar_data):
                print("[STATE] 🟧 라바콘 인식됨 → 상태 전환: CONE_NAVIGATION")
                current_mission_state = "CONE_NAVIGATION"

            # 차량 감지 조건
            if current_lidar_data is not None:
                obstacle_state = detect_vehicle_front(current_lidar_data)
                if obstacle_state == "STOP":
                        # 쿨타임 체크
                    if time.time() - last_avoidance_time < AVOIDANCE_COOLDOWN:
                        print("[⏳ 쿨타임] 최근 회피 이후 대기 중 → 차선 유지")
                        continue  # 상태 전환 안 하고 차선 유지

                    print("[STATE] 🚗 차량 감지 → 차선변경")

                    if lane_position == "Left Lane":
                        avoidance_direction = "RIGHT"
                    elif lane_position == "Right Lane":
                        avoidance_direction = "LEFT"
                    else:
                        # 차선 정보 불분명 시 기본 회피 방향 설정
                        avoidance_direction = "RIGHT"


                    # # 차선 변경 전 장애물 유무 확인
                    # if avoidance_direction == "RIGHT":
                    #     scan = current_lidar_data[225:270]
                    # else:
                    #     scan = current_lidar_data[90:130]

                    # # 10m 이내의 장애물이 5개 이상이면 회피 불가
                    # close_points = [d for d in scan if np.isfinite(d) and d < 10]
                    # print(f"[🔍 {avoidance_direction} SCAN] 총 {len(scan)}개 중 10m 이내 {len(close_points)}개")
                    # if len(close_points) >= 1000:
                    #     print(f"[❌ 회피 불가] {avoidance_direction} 차선에 장애물 있음 → 정지 또는 대기")
                    #     target_speed = 50  # 정지
                    # else:
                    print(f"[✅ 회피 가능] {avoidance_direction} 차선으로 변경 시도")
                    last_avoidance_time = time.time()
                    current_mission_state = "VEHICLE_AVOIDANCE"
                    avoidance_start_time = time.time()
                    last_avoidance_time = time.time() 
                
                elif obstacle_state == "SLOW":
                    target_speed = 50

            drive(target_angle, target_speed)
        
        elif current_mission_state == "CONE_NAVIGATION":
            # 라바콘 사이 조향각 계산
            cone_angle = calculate_cone_steering_camera(current_image_data, prev_angle, False)
            prev_angle = cone_angle
            cone_visible = detect_cones_only_camera(current_image_data)
            print(f"[STATE] 🟧 라바콘 사이 주행 중 (angle: {cone_angle:.2f})")
            target_angle = cone_angle
            target_speed = 10  # 감속 주행
            drive(target_angle, target_speed)

            if not detect_cone_presence(current_image_data, current_lidar_data) and not cone_visible:
                print("[STATE] 라바콘 감지 안됨 → LANE_FOLLOWING 복귀")
                current_mission_state = "LANE_FOLLOWING"
                cone_navigation_start_time = None
                continue

        elif current_mission_state == "VEHICLE_AVOIDANCE":
            avoidance_angle = 25 if avoidance_direction == "RIGHT" else -25
            drive(avoidance_angle, 50)

            if time.time() - avoidance_start_time > 1.5:
                print("[회피 완료] → 차선 주행 복귀")
                current_mission_state = "LANE_FOLLOWING"
                
                
        # LiDAR 거리 정보 디버깅 출력 (20도 간격)
        # if current_lidar_data is not None:
        #     bin_size = 20
        #     total_bins = 360 // bin_size
        #     print("📡 LiDAR 20도 단위 감지 개수:", end=' ')
        #     for i in range(total_bins):
        #         start = i * bin_size
        #         end = start + bin_size
        #         segment = current_lidar_data[start:end]
        #         count = len([d for d in segment if np.isfinite(d) and d < 30])
        #         print(f"{start:03d}°~{end:03d}°:{count:2d}", end=' | ')
        #     print()

        if current_lidar_data is not None:
            angles_rad = np.linspace(0, 2*np.pi, len(current_lidar_data))
            valid_indices = np.isfinite(current_lidar_data) & (current_lidar_data < 10)
            valid_ranges = current_lidar_data[valid_indices]
            valid_angles = angles_rad[valid_indices]
            
            x_coords = -valid_ranges * np.cos(valid_angles - np.pi / 2)
            y_coords = -valid_ranges * np.sin(valid_angles - np.pi / 2)
            
            # 시각화
            # lidar_points.set_data(x_coords, y_coords)
            # fig.canvas.draw_idle()
            # plt.pause(0.01)
            
            # 값 출력
            avg_distance = np.mean(valid_ranges)
            # print(f"[INFO] Average LIDAR Distance: {avg_distance:.2f}m")
        # ========================================
        # 디버깅을 위한 이미지 및 라이다 시각화
        # ========================================
        if current_image_data.size > 0:
            cv2.putText(current_image_data, f"State: {current_mission_state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(current_image_data, f"Angle: {target_angle:.1f}, Speed: {target_speed:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(current_image_data, f"Lane Position: {lane_position}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Current View (BGR)", current_image_data) # 원본 이미지 표시 
            # gray_display = cv2.cvtColor(current_image_data, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Current View (Gray)", gray_display) # 그레이스케일 이미지 표시 
            
        if cv2.waitKey(1) & 0xFF == ord('q'): # q 키를 누르면 종료
            break
        
        loop_rate.sleep() # 루프 주기 유지

    # 종료 전 모터 정지
    drive(0,0)
    print("Program terminated. Stopping vehicle.")
    cv2.destroyAllWindows()
    if 'matplotlib' in sys.modules and plt.get_fignums():
        plt.ioff()
        plt.close(fig)


#=============================================
# 메인함수 호출
# start() 함수가 실질적인 메인함수
#=============================================
if __name__ == '__main__':
    import sys # sys 모듈 import 추가
    start()