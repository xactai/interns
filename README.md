# ### 407-Person-Detection-ReID-Video-Time-OCR.py
>>>> Install OpenVINO from the official website.



### requirements.txt :

>>>> python3 -m pip install requirements.txt



### run-program:

>>>> python3 407-Person-Detection-ReID-Video-Time-OCR.py



* The Lines and Centroids are in the funtion [ draw_boxes ] in 407-Person-Detection-ReID-Video-Time-OCR.py

### Lines :

* Y-ref is the reference line in the video [Yellow line]
>>>>  Here Y- ref is the reference line for the footroid which the value is 290

* Y-up is the the [pink line]which is placed above the reference line 
>>>> Here we can count the people who are go away from the camara (i.e OUT) 
     The value of Y-up is Y-ref+60
     
* Y-down is the the [pink line] which is placed below the reference line 
>>>> Here we can count the people who are come to the camara (i.e IN) 
     The value of Y-up is Y-ref-60



### Centroids :

* Red : Which is appreaingthe person in between above Y-up(pink)line and top of the video

* Blue : Which is appreaing the person in between Y-up(pink)line and Y-ref(yellow)line

* Black : Which is appreaing the person in between Y-down(pink)line and Y-ref(yellow)line

* Green : Which is appreaingthe person in between below Y-down(pink)line and bottum of the video



### Inputs :

* Video : You can give the entire path of the video in [video_file] value

* RTSP : You can give the entire link of the RTSP streem in [video_file] value



### Time_Filter.py :

>>>> This file  which is used for detecting the time-stamp in the video/image 
>>>> It is only show the outputin the format of text(ie uncomment [ print(txts) ]  in Time_filter.py)



### 407-Person-Detection-ReID-Video-Time-OCR.py

* run_person_tracking :

>>>> This is the main function which is useed to run the program.
>>>> source is input of  function 
>>>> flip is used to rotate the video in 180 Degrees.
>>>> The use_popup is true we can see the output in the popup window  if it is false we can't see the output video only results prin5ted 	    in the terminal.



