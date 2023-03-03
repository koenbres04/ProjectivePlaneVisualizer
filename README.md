# ProjectivePlaneVisualizer

A visualization of the projective plane.
You can draw lines and points at runtime and add algebraic curves in code.
Then you can move around the affine chart from which you are looking at these objects in the projective plane, or look at the picture in the dual projective plane.
It just so happens the math works out so that by drawing the right pictures and using the right `plane_distance` (see later), one can obtain two different ways to visualize
the projective plane:
The 'sphere but glue opposite points' point of view and the 'standing on a plane and adding points at the horizon' point of view.

## getting started

As always run `git clone https://github.com/koenbres04/ProjectivePlaneVisualizer` in a folder you would like to clone the code to.
Make sure you have the python libraries numpy and python installed, then run the file main.py to run the visualization.
In the `main` function in this file you can uncomment examples to add them to the visualization.
Finally there is one important parameter to play with: the `plane_distance`.
This roughly tells you how extreme the projection is.
A plane distance of 10 is recommended when using the examples `touching_parabolas_example` and `nice_graph_example`.
A plane distance of 1 is recommended when using the examples `stand_on_plane_example` and `stand_inside_sphere_example`.
The `stand_inside_sphere_example` gives the appearence of standing inside a sphere with lines corresponding to great circles and points corresponding to antipodal pairs of points.
In the `stand_on_plane_example` gives the appearence of standing on a plane with the green line being the horizon.
During runtime the plane_distance can effectively be changed by zooming with and without holding SHIFT (see the controls section).

## controls

The visualization can be in different states and you can switch between these states using the following controls:

- ESCAPE: this puts the visualizer into the 'moving' state. When in this state the user can press
	- D to take the dual of all the lines and points in the plane. 
	This tends to put most objects close to the line at infinity, so it is recommended to have a large FOV when using this (see the bottom left corner).
- P: this puts the visualizer in the 'point drawing' state.
- ...

## how it works

...

## dependencies

The packages numpy and pygame are required.
Due to the use of some later type hinting features the code requires python 3.9 or later.
