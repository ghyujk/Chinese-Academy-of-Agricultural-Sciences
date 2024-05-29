# 读取视频，对视频进行滤波，再分别求其掩膜、膨胀，再求其交集，不能把交界线滤除或者腐蚀掉
import cv2      #需要再单目的基础上添加双目并改参数
import numpy as np
import imutils

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
        # print(frame.shape)
    #    if not ret:
    #        break  #如果读取完成，则退出循环

    #将左右两个平面分开
    frame_left=frame[0:720,0:960]  
    frame_right=frame[0:720,960:1920]
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
    
    # 显示图像
    # cv2.imshow('mask_green_left',mask_green_left) #草地掩膜,左右两边分别两块白色是滤波掩膜来的
    # cv2.imshow('mask_green_right',mask_green_right)
    # cv2.imshow('image_tu_dilate_right',image_tu_dilate_right) #草地腐蚀后的图像
    # cv2.imshow('image_green_dilate_right',image_green_dilate_right)  #草地膨胀后的图像  
    # cv2.imshow('green_filtered',green_filtered)  #掩膜的滤波
    # cv2.imshow('mask_tu_left',mask_tu_left) #土地掩膜
    # cv2.imshow('mask_tu_right',mask_tu_right)
    # cv2.imshow('image_tu_erode',image_tu_erode) #土地腐蚀后的图像
    # cv2.imshow('image_tu_dilate',image_tu_dilate) #土地先腐蚀再膨胀后的图像
    # cv2.imshow('tu_filtered',tu_filtered)   #土地掩膜的滤波
    # cv2.imshow('image_filtered',image_filtered)   #滤波后的图像
    # cv2.imshow('mask_left',overlap_mask_left) #交集
    # cv2.imshow('mask_right',overlap_mask_right)
    cv2.imshow('frame_left',frame_left)
    cv2.imshow('frame_right',frame_right)
#     cv2.imshow('frame',frame)
#     out_left.write(frame_left)
#     out_right.write(frame_right)
#     if cv2.waitKey(1) & 0xFF ==ord('q'): 
#        break
#     cv2.imshow("frame_right",frame_right)
# cap.release()
# cv2.destroyAllWindows()
    yield(frame_left,frame_right)



if __name__ == '__main__':
    video_path=r"D:\from C\Desktop\1\output2.avi"
    cap=cv2.VideoCapture(video_path)
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        process_video_frame(frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'): 
            break
        # cv2.imshow("frame_right",frame_right)
    cap.release()
    cv2.destroyAllWindows()