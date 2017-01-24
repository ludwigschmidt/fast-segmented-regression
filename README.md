# Fast Segmented Regression

This repository contains the algorithm and experiments for the following paper: 

> [Fast Algorithms for Segmented Regression](http://jmlr.org/proceedings/papers/v48/acharya16.pdf)  
> Jayadev Acharya, Ilias Diakonikolas, Jerry Li, Ludwig Schmidt  
> ICML 2016


The code is written in Julia (v0.5).

For examples of how to use the code, see the Jupyter notebooks
[`examples/histogram_example.ipynb`](https://github.com/ludwigschmidt/fast-segmented-regression/blob/master/examples/histogram_example.ipynb)
and [`examples/piecewise_linear_example.ipynb`](https://github.com/ludwigschmidt/fast-segmented-regression/blob/master/examples/piecewise_linear_example.ipynb).

The notebooks [`examples/histogram_experiments.ipynb`](https://github.com/ludwigschmidt/fast-segmented-regression/blob/master/examples/histogram_experiments.ipynb)
and [`examples/piecewise_linaer_experiments.ipynb`](https://github.com/ludwigschmidt/fast-segmented-regression/blob/master/examples/piecewise_linear_experiments.ipynb)
contain some of the experiments in the paper.
A word of caution: these notebooks were written with an earlier version of Julia and might not work out of the box at the moment.
The example notebooks above have been migrated to Julia v0.5 however.

If you want to use the code from Python, the [pyjulia](https://github.com/JuliaPy/pyjulia) package might be helpful.
