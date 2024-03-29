# Environmental Length Scales and the Galaxy-Halo Connection
An exploration of galaxy environments using explainable AI and graph neural networks

## Data

We make use of the `SUBFIND` snapshot 99 (redshift 0) subhalo catalogs derived from the Illustris TNG300-1 hydrodynamic and dark matter only (DMO) simulations. All data can be accessed through the [IllustrisTNG website](https://www.tng-project.org/data/).

## Requirements

The most important requirements are `pytorch` and `pytorch-geometric`; check out the [latter's documentation](https://pytorch-geometric.readthedocs.io/en/latest/) for more information about installing it.

Training and interpreting the explainable boosting machine (EBM) models requires the [`interpret`](https://github.com/interpretml/interpret/) and [`shap`](https://github.com/shap/shap) packages.

## Code

All of the GNN code is contained in `./src`, and the full results can be found in `./notebook/results.ipynb`. For example, if you want to train a GNN with multi-aggregation no self-loops to map dark matter only subhalo properties to galaxy stellar masses, then run:

```bash
python src/painting_galaxies.py --aggr multi --loops 0 --mode dmo
```

Training the EBM is fairly straightforward, and you can see examples in the notebook, e.g.:

```python
ebm_hyperparams = {
    "max_bins": 50000, 
    "validation_size": 0.3,
    "interactions": 32,
}

ebm = ExplainableBoostingRegressor(**ebm_hyperparams)        
ebm.fit(X_train, y_train)

y_pred = ebm.predict(X_valid)
```

## Citation 

Our paper has now been submitted to ApJ! If you use this code, please cite:
```
@ARTICLE{2024arXiv240207995W,
       author = {{Wu}, John F. and {Kragh Jespersen}, Christian and {Wechsler}, Risa H.},
        title = "{How the Galaxy-Halo Connection Depends on Large-Scale Environment}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = feb,
          eid = {arXiv:2402.07995},
        pages = {arXiv:2402.07995},
archivePrefix = {arXiv},
       eprint = {2402.07995},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240207995W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


``` 

## Acknowledgments

This project was made possible by the KITP Program, [*Building a Physical Understanding of Galaxy Evolution with Data-driven Astronomy*
](https://www.kitp.ucsb.edu/activities/galevo23) (see also the [website](https://datadrivengalaxyevolution.github.io/)).
