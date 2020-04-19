# serialdependence_radiology
This repository contains the analysis code for the project serial dependence in medical image perception.

Cristina Ghirardo
Teresa Canas-Bajo


# To do:

1. Frames in which < 10 tags are detected:
	- Right now if it's not the first frame in trial, it takes the -1 coordinates. This raises the issue that if -1 didn't find it either, maybe there are multiple NaNs.
	- For the first frame in trial, right now it's just adding NaNs. We will have to shift it to copy the +1 row. 

2. Fix delay in annotations:

	- There seems to be a delay from the annotation timestamp to where the actual blob appears. From my estimations, the delay is of ~3-4 frames. It's not super stable, fluctuating between 1-2 frames. The duration of the blob is also not stable, of 3-4 frames. We will need to adjust for this to minimize errors