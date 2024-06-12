# RoboDIMM_Extraction
Pipeline used for the extraction of the seeing parameter using data from RoboDIMM in the Observatori del Montsec.

To run the main.py file, there needs to be a folder (ResData) inside the DIMM_Data directory with all the ".res" video files. Also, the configuration file (config.txt) must be edited with:
- directory: Path where the ResData folder is located, also the path where the results will be saved.
- folder: ResData folder name to be computed.
- overwrite: Whether to re-write the centroid position files or use the ones previously saved (if they exist).
- resol: Instrument resolution in arcsec/pixel.
- w_length: Wavelength in meters.
- D: Sub-aperture diameter in meters.
- B: Distance between sub-apertures in meters.
- rW: Windowing thresholding radius in pixels.

When running the main.py file, it finds a seeing value for each video file within the ResData folder. It creates a "ResData.res" file inside the DIMM_Data directory with the results in columns as:
- #: Video file counts.
- obstime: UT time of observation.
- alt_corr: Correction for the altitude as $cos^{3/5}(z)$.
- flong: Longitudinal seeing ($\varepsilon_l$) value corrected in arcsec.
- ftran: Transversal seeing ($\varepsilon_t$) value corrected in arcsec.
- fwhm: Global seeing computed as $\sqrt{\varepsilon_l^2+\varepsilon_t^2}$, in arcsec.
