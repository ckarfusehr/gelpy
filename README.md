## To - do before making public:

Build dashboard from background removal dashboard, by inlcuding the plot ussed to setup label positions and band widths.
allow extraction of non-normalized, raw intensity values.
remove the red boundary of the line width boxes, instead draw one line where the tick is and darken the outside areas.
Add pixel axis on the y-bottom scale for the setp gel image.
The setup function should have a keyword (remove_background=False), which, when set to true, plots the plot now plotted by remove background, but with the setup gel in the top left and the bg gel in the top right.

extract magic numbers and strings to the top of each class file, using GPT.
### Write tests

* learn what the heck good tests are
* Decide on a library to use tests
* Write tests

### restructure code to optimize user experience. Use tests regularly

### Write real README file on how to use the package
### Modernize package structure to use requirements.txt file.

### Write docstring for all functions
* Check if docs generation can be derived from the docstrings?



## To-do in general:
### Change the red lane plotting to the API function show_line_profiles
This way I do not have to guess the lane_width beforehand. THings will get easier, as the Image and LinePLot classes remain more seperate, than they are atm. Also, I can then draw the rectangles only for the sliced_lengths, further helping to show what one is doing. Plus, I can see it for every set of line_plots one wants to. Just make the default that the rectangle-gel plot is shown. Also, remove connections to show_adjusted_images and just move the whole thing into an independent function?? At the moment, the funciton si not optimal and it seems to connect classes, which is not good.

### improve setup function
### softrRefactor all the code with GPT4

* Add interactive ipython widgets to setup gel
* Add some autodetection functions which give reaonable guesses for the parameters set in the setup class.
    * Implement a function to autodetect the lane positions, based on summing them up aong the y-axis, followed by a peak detection scheme. Perhaps add another function, which calculates also the optimal lane-width, used for all lanes?




### New class for quantitative band analysis
* Setup a new class, which can be used for quantitative gel analysis. E.g. it accepts the used ladder and the applied ladder mass, calculates a calibration curve out of it, and uses this to convert the ntensities of other bands into sample mass.

### Fix bugs
self.x_label_pos = x_label_pos # A workaround. Instead extrcat positions calculation from plotting function

### Unorganized thoughts:
* implement another peak finding algorithm, whichis based on inflection point detection of a spline fitted function?
* Smooth the data considerably before detecting and determining peaks.
* add util functions to crop, splot and rotate images. So people can come in directly with their recorded images.
* when setting up the label positions and widths, instead of drawing red rectangles on top, Just set all non-selected pixels to 0.1 transparency. Then peopl can directly see what is selected.
* Include some logfile, or yaml file, which saves all the used parameters for quantitative data extraction in a seperate text file. This allows ultimate reproducability.

### Publish to pypi and conda

