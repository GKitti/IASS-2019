# IASS 2019 paper

Gidofalvy, Zsarnoczay, Katula: Design formulas to estimate the ultimate load of grid shells on square plan, submitted 12/2018 to the Journal of the International Association of Shell and Spatial Structures

Linear and nonlinear finite element analysis of grid shells with a planar square boundary and with various span, span-to-height ratio, grid density, and circular hollow cross-section sizes. With the analysis, the internal forces and the ultimate load are determined. The results are investigated based on the equivalent continuum shell theory. Based on the results, a preliminary desing method is proposed.

Guide to folders:

		Folders including Jupyter files named according to the paper subsections:
		- 3.3. Internal forces
		- 3.4. Ultimate load
		- 3.5. Imperfection sensitivity analysis
		- 4.1. Rigid lateral supports
		- 4.2. No lateral supports
		- 4.4. Effect of grid layout

		TEKNO: pyhton functions and objects utilzed by the above Jupyter files. 

In the Jupyter files you can either rerun the analyis (Analysis section), or you can load the saved results (Load Results section). In the Jupyter file names, HV indicates grid shells with horizontal and vertical supports, while V indicates grid shells with only vertical supports.

All finite element analysis are performed using OpenSeesPy: https://openseespydoc.readthedocs.io/en/latest/index.html
