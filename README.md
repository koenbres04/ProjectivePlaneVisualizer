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

- ESCAPE: this puts the visualizer in the 'moving' state.
This is the default state.
In this state you can click and drag to move around the plane and scroll to zoom in and out.
Moving corresponds to rotating the plane of projection around the origin
and zooming corresponds to changing the scale with which we map this plane to the screen. (See the 'details of the projection' section for what this means.)
You can also hold SHIFT while moving and zooming.
When doing this the transformation is applied as an affine transformation on the current affine chart.
Know that SHIFT-moving and zooming may heavily distort the objects in the space.
The 'affine squishing mode' can be used to reverse this.
When in the 'moving' state you can press
	- D to take the dual of all the lines and points in the plane. 
	This tends to put most objects close to the line at infinity, so it is recommended to have a large FOV when using this (see the bottom left corner).
	- R to reset the current transformation.
- P: This puts the visualizer in the 'point drawing' state.
Click on the screen to draw points.
If you click on a line it snaps to place a point exactly on this line.
- L: This puts the visualizer in the 'line drawing' state.
Click at two points on the screen to draw a line between them.
If you click on a point the line snaps to this point.
- I: This puts the visualizer in the 'intersection drawing' state.
Click on two lines to add their intersection point to the screen.
- DELETE: This puts the visualizer in the 'deleting' state.
Click on objects to delete them.
If you click on a point on a line, it will prioritize the point.
- Y: This puts the visualizer in the 'line to infinity' state.
Click on a line to start an animation that gradually moves this line to be the line at infinity.
- A: This puts the visualizer in the 'affine squishing' state.
First click somewhere on the screen, this point will be the marked with a cross.
Now click and drag another point on the screen away or to the cross.
This will scale the plane along the line between the cross and the second clicked position.
Like SHIFT moving and zooming, this is only an affine transformation of the plane and will leave the (current) line at infinity in place.

## details of the projection

In the code, a point in the projective plane is stored as a 1-dimensional subspace of `R^3`.
To project a point to the screen, we take the intersection of the corresponding subspace with the `z=1`-plane.
The value `pixel_scale` determines at what scale these intersection points are mapped to the screen.
The `pixel_scale` also determines the FOV seen on the bottom left of the screen.
This is the angle that the screen covers when the projection plane is projected on the unit sphere.
Moving, SHIFT moving and SHIFT zooming simply perform a linear transformation of `R^3` on the projection plane.
In the code for convenience the inverse of this transformation is stored in the attribute `inv_transform` as a 3x3-matrix.
Regular moving only multiplies `inv_transform` by rotations, so it does not distort the space much.


## dependencies

The packages numpy and pygame are required.
Due to the use of some newer type hinting features the code requires python 3.9 or later.
