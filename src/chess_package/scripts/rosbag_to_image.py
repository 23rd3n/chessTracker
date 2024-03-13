#!/usr/bin/env python3 
import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayDimension
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2 

queue_size =500
coupling_delay = .035



class Ros2ImManager:

    def __init__(self) -> None:
        rospy.init_node("ros2im")

        self.pub =rospy.Publisher("/im", UInt16MultiArray, queue_size=10)
        self.sub = rospy.Subscriber ("/camera/C138422075916/color/image_raw", Image, callback=self.pose_callback)
        rospy.loginfo("Node has been started.")

        self.br = CvBridge()
        self.i=0
        self.frame = None

        self.display_thread = threading.Thread(target=self.display_camera)
        self.display_thread.start()


    def pose_callback(self, image1: Image):
        milsec_st =  1000*time.perf_counter()
        dimString = MultiArrayDimension()
        cmd = UInt16MultiArray() 
        self.frame = self.br.imgmsg_to_cv2(image1)
        if self.i % 5 == 0: 
            cmd.data = list(image1.data)
            cmd.data.append(self.i)
            dimString.label = str(int(1000*time.perf_counter() - milsec_st)) 
            cmd.layout.dim = [dimString]
            self.pub.publish(cmd)
        self.i += 1


    def display_camera(self):
        while True:
            if self.frame is not None:
                cf_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                cf_rgb = cf_rgb[150:550, 420:900]
                cv2.imshow("camera", cf_rgb)
                cv2.waitKey(1)


if __name__ == '__main__':
    
    im2yoloman = Ros2ImManager()
    rospy.spin()
    rospy.signal_shutdown("Gule Gule")
    im2yoloman.display_thread.join()
    cv2.destroyAllWindows()

