import argparse
import cv2
from object_detector import *
import agentspeak
import numpy as np
from spade import quit_spade

from spade_bdi.bdi import BDIAgent

class MyCustomBDIAgent(BDIAgent):
    def add_custom_actions(self, actions):
        @actions.add_function(".my_function", (int,))
        def _my_function(x):
            img=cv2.imread("phone.jpg")
            detector = HomogeneousBgDetector()
            contours=detector.detect_objects(img)
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (x,y),(w,h),angle=rect
                box = cv2.boxPoints(rect)
                box=np.int0(box) 
                cv2.circle(img,(int(x),int(y)),5,(0,0,255),-1)
                cv2.polylines(img,[box],True,(255,0,0),2)
                cv2.putText(img,"width {}".format(w),(int(x)+100,int(y)-100),cv2.FONT_HERSHEY_PLAIN,1, (100, 200, 0), 2)
                cv2.putText(img,"Height {}".format(h),(int(x)-100,int(y)+200),cv2.FONT_HERSHEY_PLAIN,1, (100, 200, 0), 2)
            cv2.imshow('image',img)
            cv2.waitKey(0)
            return x * x

        @actions.add(".my_action", 1)
        def _my_action(agent, term, intention):
            arg = agentspeak.grounded(term.args[0], intention.scope)
            print(arg)
            yield


a = MyCustomBDIAgent("hassine@desktop-ahuhkk8", "admin", "actions.asl")

a.start()

import time

time.sleep(2)
a.stop().result()

quit_spade()