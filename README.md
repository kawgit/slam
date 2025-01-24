# SLAM

I've created an extremely rudimentary implementation of a simultaneous localization and mapping (SLAM) system. By running the main.py script (if left on the default settings in settings.py) you should see a point cloud appear which has been created by tracking the movement of points in a video of a landscape as seen by a person riding in a moving car.

## Limitations
- Fails to work if the camera is rotating. The camera can move through a scene but must always face the same direction.
- No way to determine absolute distances, instead it can only determine relative distances between points. 
- No points persist after they disappear from the screen. The point cloud only depicts the immediate contents of the frame, nothing more. 