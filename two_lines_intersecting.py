# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:33:46 2020

@author: juru
"""

import numpy as np

def two_lines_intersecting(line1,line2):
    """
    """
    intersect = False
    if (((line1[0][0] == line1[1][0]) and (line1[0][1] == line1[1][1])) or ((line2[0][0] == line2[1][0]) and (line2[0][1] == line2[1][1]))):
        intersect = False
    else:
        x1 = [line1[0][0],line1[1][0]]
        y1 = [line1[0][1],line1[1][1]]
    #    plt.plot(x1, y1, label = "line 1")  
        x2 = [line2[0][0],line2[1][0]]
        y2 = [line2[0][1],line2[1][1]]
    #    plt.plot(x2, y2, label = "line 2")   
    
        if (line1[1,0] - line1[0,0])!=0:
            m1=(line1[1,1] - line1[0,1])/(line1[1,0] - line1[0,0])
        else:
            if (line1[1,1] - line1[0,1])>0:
                m1=float('inf')
            else:
                m1=float('-inf')
        if (line2[1,0] - line2[0,0])!=0:
            m2=(line2[1,1] - line2[0,1])/(line2[1,0] - line2[0,0])
        else:
            if (line2[1,1] - line2[0,1])>0:
                m2=float('inf')
            else:
                m2=float('-inf')                        
        #m1 = np.true_divide((line1[1,1] - line1[0,1]), (line1[1,0] - line1[0,0]))
        #m2 = np.true_divide((line2[1,1] - line2[0,1]), (line2[1,0] - line2[0,0]))
        b1 = line1[0,1] - m1*line1[0,0]
        b2 = line2[0,1] - m2*line2[0,0]
        check_val=False
        if (((m1 != np.inf)and(m1 != -np.inf)) and ((m2 != np.inf)and(m2 != -np.inf))):
            check_val=True
            if (m1-m2)!=0:
                xintersect=(b2-b1)/(m1-m2)
            else:
                if (b2-b1)>0:
                    xintersect=float('inf')
                else:
                    xintersect=float('-inf')                         
        #xintersect = np.true_divide((b2-b1), (m1-m2))
        #yintersect = m1*xintersect + b1
        
        if ((np.abs(m1-m2)>1e-6) and (check_val==True)):
            isPointInsidex1 = (
                ((xintersect - line1[0,0]) > 1e-6 and (xintersect - line1[1,0]) < -1e-6 ) or 
                ((xintersect - line1[1,0]) > 1e-6 and (xintersect - line1[0,0]) < -1e-6))
        
            isPointInsidex2 = (
                ((xintersect - line2[0,0]) > 1e-6 and (xintersect - line2[1,0]) < -1e-6 ) or
                ((xintersect - line2[1,0]) > 1e-6 and (xintersect - line2[0,0]) < -1e-6))
        
            inside = isPointInsidex1 and isPointInsidex2
            intersect = inside
            
        
        if (np.abs(m1-m2)<1e-6) :
            if (np.abs(b1-b2)>1e-6) :
                intersect = False
                
            if (np.abs(b1-b2)<1e-6) :
                isPointInside12 = (((line1[0,0] - line2[0,0]) > 1e-6 and
                    (line1[0,0] - line2[1,0]) < -1e-6 ) or
                    ((line1[0,0] - line2[1,0]) > 1e-6 and
                    (line1[0,0] - line2[0,0]) < -1e-6))
                
                isPointInside22 = (((line1[1,0] - line2[0,0]) > 1e-6 and
                    (line1[1,0] - line2[1,0]) < -1e-6 ) or
                    ((line1[1,0]- line2[1,0]) > 1e-6 and
                    (line1[1,0] - line2[0,0]) < -1e-6))
                inside = isPointInside12 or isPointInside22
                intersect = inside
                
        
        if (((m1 == np.inf) or (m1 == -np.inf)) or ((m2 == np.inf) or (m2 == -np.inf))):
            if (((m1 != np.inf)and(m1 != -np.inf)) or ((m2 != np.inf)and(m2 != -np.inf))):
                if ((m1 != 0) and (m2 != 0)):
                    line3 = np.zeros((2,2))
                    line4 = np.zeros((2,2))
                    line3[0,0] = line1[0,1]
                    line3[0,1] = line1[0,0] 
                    line3[1,0] = line1[1,1] 
                    line3[1,1] = line1[1,0] 
                    line4[0,0] = line2[0,1] 
                    line4[0,1] = line2[0,0] 
                    line4[1,0] = line2[1,1]
                    line4[1,1] = line2[1,0]
                    m3 = (line3[1,1] - line3[0,1])/(line3[1,0] - line3[0,0])
                    m4 = (line4[1,1] - line4[0,1])/(line4[1,0] - line4[0,0])
                    b3 = line3[0,1] - m3*line3[0,0]
                    b4 = line4[0,1] - m4*line4[0,0]
                    xintersect2 = (b4-b3)/(m3-m4)
                    #yintersect2 = m3*xintersect2 + b3
                    isPointInsidex6 = (
                    ((xintersect2 - line3[0,0]) > 1e-6 and (xintersect2 - line3[1,0]) < -1e-6 ) or 
                    ((xintersect2 - line3[1,0]) > 1e-6 and (xintersect2 - line3[0,0]) < -1e-6))
                    isPointInsidex7 = (
                    ((xintersect2 - line4[0,0]) > 1e-6 and (xintersect2 - line4[1,0]) < -1e-6 ) or 
                    ((xintersect2 - line4[1,0]) > 1e-6 and (xintersect2 - line4[0,0]) < -1e-6))
        
                    inside = isPointInsidex6 and isPointInsidex7
                
                    intersect = inside
                    
                else:
                    if (m1==0):
                        y1=line1[0,1]
                        x1min=np.min((line1[0,0],line1[1,0]))
                        x1max=np.max((line1[0,0],line1[1,0]))
                        x2=line2[0,0]
                        y2min=np.min((line2[0,1],line2[1,1]))
                        y2max=np.max((line2[0,1],line2[1,1]))
                        if ((y1>y2min)and(y1<y2max)and(x2>x1min)and(x2<x1max)):                      
                            intersect = True
                            
                        else:
                            intersect = False
                            
                    if (m2==0):
                       y2=line2[0,1]
                       x2min=np.min((line2[0,0],line2[1,0]))
                       x2max=np.max((line2[0,0],line2[1,0]))
                       x1=line1[0,0]
                       y1min=np.min((line1[0,1],line1[1,1]))
                       y1max=np.max((line1[0,1],line1[1,1]))
                       if ((y2>y1min)and(y2<y1max)and(x1>x2min)and(x1<x2max))   :  
                           intersect = True
                           
                       else:
                           intersect = False
                           
            if (((m1 == np.inf)or(m1 == -np.inf)) and ((m2 == np.inf)or(m2 == -np.inf))):
                if (line1[0,0] == line2[0,0]) : 
                    insidet= (((line1[0,1] - line2[0,1]) > 1e-6  and (line1[0,1] - line2[1,1]) < -1e-6) or 
                    ((line1[0,1] - line2[1,1]) > 1e-6  and (line1[0,1] - line2[0,1]) < -1e-6))
                    insidep=(((line1[1,1] - line2[0,1]) > 1e-6  and (line1[1,1] - line2[1,1]) < -1e-6) or 
                    ((line1[1,1] - line2[1,1]) > 1e-6  and (line1[1,1] - line2[0,1]) < -1e-6))
                    inside = insidet or insidep
                    intersect = inside
                    
                if (line1[0,0] != line2[0,0]):
                    intersect = False
                    
    return intersect


if __name__ == '__main__':
    line1=np.array([[1,2],[3,4]]) # The first column represents to x-values of the line segment line[0,0] y line[1,0]. The second column represents the y-values of
                                    #the line segment
    line2=np.array([[1,4],[3,2]])
    
    line1 = np.random.rand(4).reshape((2,2))*10-5
    line2 = np.random.rand(4).reshape((2,2))*10-5
    
#    line1=np.array([[1,2],[7,2]]) 
#    line2=np.array([[5,3],[5,1]])
    
    intersect = two_lines_intersecting(line1,line2)
    print(intersect)