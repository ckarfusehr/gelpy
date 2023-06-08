## To-do:

### Improve the background removal

* make current background correction optional
* Implement new background correction, independently
    * restrict background linefit to only beginning and end of line profiles
        * Or do not do the line fit, but instead substract the background early on, even before noralizing to area under the curve, as this is currently biased by the background. more so for low intensity lanes.
* Integrate new background correction as option

### Create setup class

* extract relevant functions from other packages
    * Implement setup_gel() function of Agarose class, which allows setting the lane x axis positions. It should do some autoadjustments of the contrast to make it easier. It should also directly use the lane width and plot the used lane for line profile extraction as a slightly red overlay.
* Add interactive ipython widgets to setup gel
* Add some autodetection functions which give reaonable guesses for the parameters set in the setup class.
    * Implement a function to autodetect the lane positions, based on summing them up aong the y-axis, followed by a peak detection scheme. Perhaps add another function, which calculates also the optimal lane-width, used for all lanes?

### Write tests

* learn what the heck good tests are
* Decide on a library to use tests
* Write tests


### New class for quantitative band analysis
* Setup a new class, which can be used for quantitative gel analysis. E.g. it accepts the used ladder and the applied ladder mass, calculates a calibration curve out of it, and uses this to convert the ntensities of other bands into sample mass.

### Fix bugs

* gaussian plotting error

### Write real README file on how to use the package

### Modernize package structure to use requirements.txt file.

### Write docstring for all functions
* Check if docs generation can be derived from the docstrings?

### Unorganized thoughts:
* Setup that lanes can also be selected by label names, not only by indices
* Remove the if statements in the fit plotting rotine. perhaps by using the dataframe? Or by changing the call signature of the single_peak function, to accept a tuple of function params, instead of them individual. Then I can just pass in the params of the current line_profile
* implement another peak finding algorithm, whichis based on inflection point detection of a spline fitted function?
* Enable linear contrast adjustment to percentiles. and automatically detectif provided linear contrast adjustment is meant to be percentiles (0-1) or raw intensity values.

### Publish to pypi and conda

