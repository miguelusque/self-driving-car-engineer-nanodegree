PID values have been chosen by try and error. I thought about implementing Twiddle, but the resulting
code was not elegant enought to include in this submission.

A higher value of the proporcioanl controller (Kp_) makes the car to oscilate faster, moving the car 
from one side to the other side of the road, which can make the driver dizzy. A smaller value of Kp_
makes the oscillation slower.

The differential controller (Kd_) helps the car to  prevent overshoot. When the error is getting smaller
over time, it counter steers. It is calculated as  the difference of the current crosscheck error and
the previous one. If I set this parameter to 0, it couses a high CTE.

The integral term (Ki_) has been set to 0.0, because the objective of this parameters is to compensate for 
bias, and this car seems to do not have any bias. 
If we set it to a different value, i.e 0.5, the car crashes agains the side of the road.

Finally, please let me apologize for my English. I am not a native English speaker. 