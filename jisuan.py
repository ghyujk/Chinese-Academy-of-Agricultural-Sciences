import math
import cv2
import numpy as np
import sympy
from sympy import *
# from final2_final import process_video_frame
import imutils

violet=0.11
angle=15
cos_value=math.cos(math.radians(angle))
sin_value=math.sin(math.radians(angle))

x  = sympy.symbols('x')
y  = sympy.symbols('y')
z  = sympy.symbols('z')
a1 = sympy.symbols('a1')
b1 = sympy.symbols('b1')
# 读取视频
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"D:\from C\Desktop\1\output2.avi")


# 获取视频的帧率和尺寸
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建输出视频文件
# out_left = cv2.VideoWriter('left.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width//2, height))
# out_right = cv2.VideoWriter('right.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width//2, height))

# 计算中心点坐标
# center_x = int(width / 2)
# center_y = int(height / 2)
# center_x_half=int(center_x / 2)

# 创建视频编写器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

def process_video_frame(frame):
    # frame=video
    # 读取视频
    # video_path=r"D:\Vscode\nong\output2.avi"
    # cap=cv2.VideoCapture(video_path)

    # 存视频
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fourcc= cv2.VideoWriter_fourcc(*'XVID')
    # out_left = cv2.VideoWriter(r"D:\Vscode\ziyuan\output_left2.avi",fourcc,30, (640, 480))  # 保存视频
    # out_right = cv2.VideoWriter(r"D:\Vscode\ziyuan\output_right2.avi",fourcc,30, (640, 480))  # 保存视频

    # 定义颜色空间
    lower_green=np.array([30,58,0]) #掩膜：在阈值范围内的赋值为255（白色），其余是0（黑色）
    higher_green=np.array([179,255,255])
    lower_tu=np.array([30,0,51])  #10,50,20(灰色)  33,48,45  0,0,81(绿色和棕色)
    higher_tu=np.array([179,39,255])   #25,255,255  35,255,255  36,255,255
    # while True:
    #    ret,frame=cap.read()  #读取是视频中一帧的画面
    #     print(frame.shape)
    #    if not ret:
    #        break  #如果读取完成，则退出循环

    # 将左右两个平面分开
    frame_left=frame[0:480,0:640]  
    frame_right=frame[0:480,640:1280]
    frame_left = imutils.resize(frame_left,640)
    frame_right = imutils.resize(frame_right,640)
    # cv2.imshow("frame_right",frame_right)
    # cv2.waitKey(0)
    # frame_right = cv2.flip(frame_right, 1)
    image_filtered_left=cv2.medianBlur(frame_left,9)    #对图像进行中值滤波，第二个参数要是奇数。滤波小一点，腐蚀大一点
    image_filtered_right=cv2.medianBlur(frame_right,11)
    
    # 将滤波后的图像转换为hsv图像
    hsv_left=cv2.cvtColor(image_filtered_left,cv2.COLOR_BGR2HSV)  
    hsv_right=cv2.cvtColor(image_filtered_right,cv2.COLOR_BGR2HSV)
    
    # 分别求其掩膜
    mask_green_left=cv2.inRange(hsv_left,lower_green,higher_green) #分别对左右求绿色的掩膜
    mask_green_right=cv2.inRange(hsv_right,lower_green,higher_green)
    mask_tu_left=cv2.inRange(hsv_left,lower_tu,higher_tu) #分别对左右求土色的掩膜
    mask_tu_right=cv2.inRange(hsv_right,lower_tu,higher_tu)
    # cv2.imshow('mask_tu_right',mask_tu_right)
    # cv2.imshow('mask_green_right',mask_green_t)
    
    # 对分割的区域进行膨胀
    kernel=np.ones((2,2),np.uint8)  #定义卷积核(5,5)和(4,4)
    kernel2=np.ones((4,4),np.uint8)
    image_green_erode_left=cv2.erode(mask_green_left,kernel,iterations=10) #对左右草地掩膜腐蚀(2)
    image_green_erode_right=cv2.erode(mask_green_right,kernel,iterations=18)
    image_green_dilate_left=cv2.dilate(image_green_erode_left,kernel2,iterations=15) #对腐蚀后的草地掩膜进行膨胀
    image_green_dilate_right=cv2.dilate(image_green_erode_right,kernel2,iterations=14)
    
    image_tu_erode_left=cv2.erode(mask_tu_left,kernel,iterations=10) #对左右土地掩膜腐蚀(2)
    image_tu_erode_right=cv2.erode(mask_tu_right,kernel,iterations=18)
    image_tu_dilate_left=cv2.dilate(image_tu_erode_left,kernel2,iterations=15) #对腐蚀后的土地掩膜进行膨胀
    image_tu_dilate_right=cv2.dilate(image_tu_erode_right,kernel2,iterations=14)

    # 在交集处画出分界线
    overlap_mask_left=cv2.bitwise_and(image_green_dilate_left,image_tu_dilate_left)  #求并集
    overlap_mask_right=cv2.bitwise_and(image_green_dilate_right,image_tu_dilate_right)
    # image=cv2.erode(overlap_mask,kernel2,iterations=2) #对并集腐蚀
    # image=cv2.dilate(image,kernel,iterations=3) #膨胀
    contours_left=cv2.findContours(overlap_mask_left,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #检测轮廓
    contours_right=cv2.findContours(overlap_mask_right,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lines_left=cv2.HoughLinesP(overlap_mask_left,1,np.pi/180,100,minLineLength=150,maxLineGap=10) #霍夫直线检测cv2.HoughLinesP
    # print(lines_left)
    if lines_left is not None:
    #    for line in lines_left:
            # print(line)
        x1,y1,x2,y2=lines_left[0][0]
                # print(x1,y1,x2,y2)
        pts=np.array([[x1,y1],[x2,y2]],np.int32)
        cv2.polylines(overlap_mask_left,[pts],False,(0,0,255),2)
        cv2.polylines(frame_left,[pts],False,(0,0,255),6)  #画线
    
    lines_right=cv2.HoughLinesP(overlap_mask_right,1,np.pi/180,100,minLineLength=150,maxLineGap=10) #霍夫直线检测cv2.HoughLinesP
    if lines_right is not None:
    #    for line in lines_right:
        x3,y3,x4,y4=lines_right[0][0]
        pts=np.array([[x3,y3],[x4,y4]],np.int32)
        cv2.polylines(overlap_mask_right,[pts],False,(0,0,255),2)
        cv2.polylines(frame_right,[pts],False,(0,0,255),6)  #画线
    return frame_left,frame_right

# 解方程组
def solve_eqs(k_1, b_1, m_k_2, m_k_z, m_b_2):
    N = np.linalg.solve([[k_1, 0], [m_k_2, m_k_z]], [1-b_1, 1-m_b_2])
    return N[0], N[1]

# 计算直线上一点Q
def calc_Q(k_1, b_1, m_k_2, m_k_z, m_b_2):
    a1, b1 = solve_eqs(k_1, b_1, m_k_2, m_k_z, m_b_2)
    return np.array([a1, 1, b1])

# 计算向量w
def calc_w(k_1):
    return np.array([k_1, -1, 0]) #左相机方程系数提出来的

# 计算向量m
def calc_m(m_k_2, m_k_z):
    return np.array([m_k_2, -1, m_k_z])

# 计算相交线的方向向量q
def calc_q(k_1,  m_k_2, m_k_z):
    w = calc_w(k_1)
    m = calc_m(m_k_2, m_k_z)
    return np.cross(w, m)

# 计算方向向量的单位向量
def normalize_vector(vector):
    length = np.linalg.norm(vector)
    return vector / length

# 计算相交线单位向量与X轴方向的单位向量的夹角
def angle_with_z_axis(vector):
    unit_vector = normalize_vector(vector)
    z_unit_vector = np.array([0, 0, 1])  # Z轴方向的单位向量
    dot_product = np.dot(unit_vector, z_unit_vector)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg



# def normalize_vector(vector):
#     length = np.linalg.norm(vector)
#     return vector / length




# 计算向量Op，即原点到直线上一点的方向向量
def calc_p(k_1, b_1, m_k_2, m_k_z, m_b_2):
    Q = calc_Q(k_1, b_1, m_k_2, m_k_z, m_b_2)
    return -Q


# 计算向量l（直线方向向量与原点指向Q点所构成的向量的叉乘的绝对值，再除以方向向量的模）
def calc_l(k_1, b_1, m_k_2, m_k_z, m_b_2):
    Q = calc_Q(k_1, b_1, m_k_2, m_k_z, m_b_2)
    w = calc_w(k_1)
    m = calc_m(m_k_2, m_k_z)
    q = calc_q(k_1,  m_k_2, m_k_z)#方向向量
    p = calc_p(k_1, b_1, m_k_2, m_k_z, m_b_2)#原点指向Q点所构成的向量
    l = np.cross(p, q)
    return math.sqrt(np.sum(l**2)) / math.sqrt(np.sum(q**2))

# while cap.isOpened():
#    ret, frame = cap.read()
#    if not ret:
#        break
while true:
# for   value in  final1_final.process_video_frame(path)  :
    # left_frame,right_frame = value  
    ret,frame = cap.read()
    left_frame,right_frame  =  process_video_frame(frame)
    # right_frame  =  frame_right
    height=left_frame.shape[0]
    width=left_frame.shape[1]
    # 在视频中心创建x和y坐标轴
    cv2.line(left_frame, (width, 0), (width,height), (0, 255, 0), 1)#左相机的横轴
    cv2.line(left_frame, (0, height), (width, height), (0, 255, 0), 1)#左相机的竖轴
    cv2.line(right_frame, (width, 0), (width, height), (0, 255, 0), 1)#右相机的横轴
    cv2.line(right_frame, (0, height), (width, height), (0, 255, 0), 1)#右相机的竖轴

    # Convert frame to HSV color space
    hsv_1 = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
    hsv_2 = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([0,50,50]) 
    upper_red = np.array([10,255,255])

    mask1_1 = cv2.inRange(hsv_1, lower_red, upper_red)
    mask1_2 = cv2.inRange(hsv_2, lower_red, upper_red)

    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])

    mask2_1 = cv2.inRange(hsv_1, lower_red, upper_red)
    mask2_2 = cv2.inRange(hsv_2, lower_red, upper_red)

    # Combine masks
    mask_1 = cv2.bitwise_or(mask1_1, mask2_1)
    mask_2 = cv2.bitwise_or(mask1_2, mask2_2)

    kernel = np.ones((5, 5), np.uint8)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, kernel)
    mask_2 = cv2.morphologyEx(mask_2, cv2.MORPH_OPEN, kernel)


    dst_1 = cv2.equalizeHist(mask_1)
    dst_2 = cv2.equalizeHist(mask_2)

    gaussian_1 = cv2.GaussianBlur(dst_1, (9, 9), 0)
    gaussian_2 = cv2.GaussianBlur(dst_2, (9, 9), 0)
    # 高斯滤波降噪

    # Apply Canny edge detection
    edges_1 = cv2.Canny(gaussian_1, 70, 150)
    edges_2 = cv2.Canny(gaussian_2, 70, 150)

    # 进行霍夫变换
    lines_1 = cv2.HoughLines(edges_1,2,np.pi/180,200)
    lines_2 = cv2.HoughLines(edges_2,2,np.pi/180,200)

    # Draw lines on frame
    if lines_2 is not None and lines_1 is not None:
        for line_1 in lines_1[0]:  #可以先轮廓逼近。然后检测单边轮廓
            rho_1,theta_1 = line_1
            a_1 = np.cos(theta_1)
            b_1 = np.sin(theta_1)
            x0_1 = a_1*rho_1
            y0_1 = b_1*rho_1
            x1_1 = int(x0_1 + 1000*(-b_1))
            y1_1 = int(y0_1 + 1000*(a_1))
            x2_1 = int(x0_1 - 1000*(-b_1))
            y2_1 = int(y0_1 - 1000*(a_1))
            cv2.line(left_frame,(x1_1,y1_1),(x2_1,y2_1),(0,255,255),2)
            #print(x1_1,y1_1)

    # Draw lines on frame

        for line_2 in lines_2[0]:  #可以先轮廓逼近。然后检测单边轮廓
            rho,theta = line_2
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(right_frame,(x1,y1),(x2,y2),(255,255,255),2)

    # 两个方程的原点在左上角，其中x轴向右延伸，y轴向下延伸



    if x1_1==x2_1: print("直线斜率不存在，方程y=",x1_1)
    else : 
        k_1=(y1_1-y2_1)/(x1_1-x2_1)
        b_1=y1_1-k_1*x1_1
        y_1=k_1*x+b_1
        # print("y1=",y_1)



    if x1==x2:print("直线斜率不存在，方程y=",x1)
    else :
        k_2=(y1-y2)/(x1-x2)
        b_2=y1-k_2*x1
        y_2=k_2*x+b_2
        m_k_2=k_2*cos_value
        m_k_z=-1*k_2*sin_value
        m_b_2=k_2*violet*cos_value+b_2
        # print("y2=",y_2)

        g = calc_l(k_1, b_1, m_k_2, m_k_z, m_b_2)
        vector = calc_q(k_1,  m_k_2, m_k_z)
        line_vector = normalize_vector(vector)
        projection_vector = np.array([line_vector[0], 0, line_vector[2]])
        angle_with_z = angle_with_z_axis(projection_vector)
        # h = 180 - angle_with_x_axis(vector)
        print("距离：",g,"角度：",angle_with_z)


        cv2.imshow('left',left_frame)
        cv2.imshow('right',right_frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'): 
            break



 