# Welcome to the gelpy project

If you are looking for a Python package for simplified, reproducible, and easily shareable gel electrophoresis analysis, you are in the right place. If you are also keen on contributing to a Python project, then hey, you are also in the right place!

Please note that this package is under development and currently in its alpha phase. I would like to invite you to **try gelpy, break it, and contribute to it**. Valuable contributions include raising issues on the GitHub issues page or, if you want to do some coding, fixing the issue and submitting a merge request to this repository. In any case, I believe that even in its early state, gelpy can make your life working with electrophoresis gels more pleasant and reproducible.

## What can gelpy do?

As of now, this package includes functions to:

- Visualize gels with automatically adjusted contrasts (linear and non-linear)
- Label the lanes
- Remove the image background
- Extract and plot lane profiles
- Fit the peaks (bands) of lane profiles
- Save all generated images
- Save the gel object, including all the plots, lane profiles, and fits, as a Python object. You can then send this object as one file to your collaborator, who can load it back into Python and continue where you left off.

Another benefit of using this package over, for example, Fiji for data extraction is that all lane profiles, fitting parameters, and the original image are directly available in Python and do not have to be saved and imported again for further custom analysis

## Installation

### pip

```
conda create -n <your_environment_name>
conda install python
pip installl gelpy

```

(`conda install python`is only needed if your system wide python installation is older than 3.8. This is because `conda create -n <env_name>` gives you an empty environment without Python in it):


## How to use gelpy

I recommend checking out the example Jupyter notebook in the "Example" folder. I hope to add proper documentation soon. Also, please note that the docstring was generated with GPT and may potentially contain errors in its current state.

## How to contribute by coding

Does this pique your interest in contributing? Perhaps even implementing new features? I have an unstructured list below where I collect ideas for tasks to be done or features to be implemented.

### Loose notes

- translate docstrings to reStructuredText, such that Sphinx can read it.
- Compose docs with Sphinx and make it available on read the docs.
- Allow direct extraction of non-normalized, raw intensity values.
- Extract remaining magic numbers and magic strings to the top of each class file (using GPT).
- Write (more) unit tests.
- Add optional interactive IPython widgets to set up the gel.
- Add some autodetection functions that provide reasonable guesses for the parameters in the `gel.setup_gel()` function.
  - Implement a function to autodetect the lane positions based on summing them up along the y-axis, followed by a peak detection scheme. Perhaps add another function that calculates the optimal lane width used for all lanes?
- Implement another peak-finding algorithm based on inflection point detection of a spline-fitted function.
- Add utility functions to crop and rotate images so that people can directly use their recorded images.
- Include a logfile or YAML file that saves all the used parameters for quantitative data extraction in a separate text file. This simplifies reproducibility.

### New class for quantitative band analysis

- Set up a new class that can be used for quantitative gel analysis (e.g., in ng per band). The class should accept the ladder used, the applied ladder concentration, and volume. It should calculate a calibration curve based on this information and use it to convert the intensities of other bands into sample mass.