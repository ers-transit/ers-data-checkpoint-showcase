# Test Notebooks for `ers-transit` Data Checkpoints

This repository contains a collection of utilities and notebooks related to transiting exoplanet data from the James Webb Space Telescope. The notebooks are designed to provide uniform(-ish) visualizations and metrics at various checkpoints along the data analysis process, to ease comparison between different independent analyses.

![visual summary of data analysis checkpoints for transiting exoplanet atmosphere data](https://ers-transit.github.io/images/summary-of-ers-transit-steps.png)

Some key checkpoints covered by these notebooks include:
- **Calibrated 2D pixel images.** A lot of processing needs to happen before getting to an image that estimates the rate at which photons hit each pixel on the detector. The notebook(s) in `images/` help visualize and evaluate these images, both before and after background subtraction.
- **Extracted 1D stellar spectra.** Once a time-series collection of extracted spectra has been assembled, we can start looking at the noise and variability properties of across both wavelength and time. The notebook(s) in `spectra/` help visualize and evaluate these time-series spectra.
- **Fitted transit features.** By fitting models to the time-series measurements of the wavelength-dependent flux from the system, we can infer properties of the transiting planet, like its transit depth $(R_p/R_\star)^2$ or eclipse depth $(F_p/F_\star)$. The notebook(s) in `features/` help visualizat and evaluate these fitted planetary features.
- *(coming soon)* **Binned multi-wavelength light curves and residuals.** When fitting for planetary features, there are lots of different choices you can make or methods you can use to characterize systematic noise sources. The notebook(s) in `lightcurves/` *will* help visualize and evaluate some aspects of your fits by looking at your binned light curves and their residuals compared to models.

As of March 8, 2022 the notebooks in these folders are only snippets to make sure that the data results you submit can be read in with a recognizable format. More detailed notebooks will be made available and continue to be revised throughout the March 21-23, 2022 meetings of the [ers-transit Data Challenge with Simulated Data](https://ers-transit.github.io/data-challenge-with-simulated-data.html). 
