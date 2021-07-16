# Feature miner

1. Clean the raw articles
```bash
python clean_articles.py
```

2. Build the feature matrices
```bash
python create_feature_matrices.py
```

3. Do random grid searches
```bash
python Grid_search/leaveoneout/rf_random_search.py
python Grid_search/leaveoneout/svm_random_search.py
python Grid_search/onevsone/rf_random_search.py
```

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
```bash
python make_plots.py
python calculate_correlations.py
```

6. When writing paper
```bash
python prepare_metadata.py
```