[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Continent
Code for our KDD2024 paper titled [Learning causal networks from episodic data](https://dl.acm.org/doi/abs/10.1145/3637528.3671999)

> Tldr: We discover causal graphs for data obtained in chunks over time. We do not force a single causal network but rather cluster these chunks into multiple potential causal structures.


Use **`main.py`** to see an example of how the code works.

----------------------------------------------------------
REQUIREMENTS:
----------------------------------------------------------
- Python version: 3.8 or better
- Packages:
	* scikit-learn
	* numpy
	* mpmath
	* matplotlib
	* cdt (causal discovery toolbox)
  * rpy2 (to run Rcode of earth regression spline package)  

- R Version: 3.6 or better
- R packages:
	* SID: 	 https://www.rdocumentation.org/packages/SID/versions/1.0
	* pcalg: https://www.rdocumentation.org/packages/pcalg/versions/2.6-8
	* kpcalg/RCIT:  https://github.com/Diviyan-Kalainathan/RCIT
	* spresebn: https://www.rdocumentation.org/packages/sparsebn/versions/0.1.0
	* bnlearn: https://www.rdocumentation.org/packages/bnlearn/versions/4.5
    * earth: https://cran.r-project.org/package=earth


## Contact
For errors and corrections, please reach out via https://sites.google.com/view/mian.
