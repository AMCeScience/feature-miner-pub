# Feature miner

Developed and tested on Windows 10 inside a `venv` environment using Python 3.7.7 and pip 19.2.3

Setup your environment by installing the requirements using pip.
```bash
pip install -r requirements.txt
```

Copy the config.example file to `config.py`. A database with dummy data is provided [here](https://figshare.com/s/165be05a58e9ec3553ec), place the file in `/Database/miner_database.db`. Data used in the research is available upon request (contact: [a.j.vanaltena@amsterdamumc.nl](mailto:a.j.vanaltena@amsterdamumc.nl)) or may be collected from PubMed using the `qrel` files from the [2017 CLEF eHealth Lab](https://github.com/CLEF-TAR/tar/tree/master/2017-TAR). Follow the steps below to perform the experiments.

### Setup

1. Clean the raw articles
```bash
python clean_articles.py
```

2. Build the feature matrices
```bash
python create_feature_matrices.py
```

3. Do grid searches
```bash
python Grid_search/leaveoneout/rf_random_search.py
python Grid_search/onevsone/rf_random_search.py
```

*Note:* the results of the grid searches are placed in a csv file in the `Grid_search/leaveoneout/` and `Grid_search/onevsone/` directories respectively.

### Experiments

Create a folder with the name of the experiment run and edit the `CLASSIFIER_LOCATION` in the config.py file. The [config.example](config.example) file uses the foldername `run1`.

4. Run the classifiers
```bash
python run_leaveoneout.py
python run_onevsone.py
python run_nvsone.py
python run_nvsone_random.py

# Fetch timing difference results for two training set sizes
python run_nvsone_timing.py
```

5. Interpret the outcomes

*Note:* for correlations calculation a metadata file is necessary. You may find this file for the fifty reviews used in our research [here](https://doi.org/10.6084/m9.figshare.7804094.v1). For testing purposes we also provide a [dummy set](https://figshare.com/s/9d2228ef5170773e3aa1).
```bash
python make_plots.py
python calculate_correlations.py
```

6. When writing paper
```bash
python prepare_metadata.py
```