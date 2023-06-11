## To - do before making public:

* make current background correction optional
* Implement new background correction, independently
    * restrict background linefit to only beginning and end of line profiles
        * Or do not do the line fit, but instead substract the background early on, even before noralizing to area under the curve, as this is currently biased by the background. more so for low intensity lanes.
* Integrate new background correction as option
### Write real README file on how to use the package
### Modernize package structure to use requirements.txt file.

### Write docstring for all functions
* Check if docs generation can be derived from the docstrings?
### Improve the background removal




## To-do in general:

### Create setup class

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
self.x_label_pos = x_label_pos # A workaround. Instead extrcat positions calculation from plotting function

### Unorganized thoughts:
* implement another peak finding algorithm, whichis based on inflection point detection of a spline fitted function?

### Publish to pypi and conda

