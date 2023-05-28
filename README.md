### To-do:

* restrict background linefit to only beginning and end of line profiles
    * Or do not do the line fit, but instead substract the background early on, even before noralizing to area under the curve, as this is currently biased by the background. more so for low intensity lanes.
* Remove the if statements in the fit plotting rotine. perhaps by using the dataframe? Or by changing the call signature of the single_peak function, to accept a tuple of function params, instead of them individual. Then I can just pass in the params of the current line_profile
* Implement setup_gel() function of Agarose class, which allows setting the lane x axis positions. It should do some autoadjustments of the contrast to make it easier. It should also directly use the lane width and plot the used lane for line profile extraction as a slightly red overlay.
* implement another peak finding algorithm, whichis based on inflection point detection of a spline fitted function?
* Implement a function to autodetect the lane positions, based on summing them up aong the y-axis, followed by a peak detection scheme. Perhaps add another function, which calculates also the optimal lane-width, used for all lanes?
* Setup a new class, which can be used for quantitative gel analysis. E.g. it accepts the used ladder and the applied ladder mass, calculates a calibration curve out of it, and uses this to convert the ntensities of other bands into sample mass.
* Setup that lanes can also be selected by label names, not only by indices
* Enable linear contrast adjustment to percentiles.
* Automatically detectif provided linear contrast adjustment is meant to be percentiles (0-1) or raw intensity values.
* For the EMG function, the fitted mean should not be restricted so tightly to the maxima index, as for strong skewedness, the mean of the EMG is not at it's maxima.