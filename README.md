Files and quick start

```
demo_data.csv   800 cases + 800 controls simulated from Scenario 1 of the paper; columns id, Y, X1, X2, X3
m01.R  m02.R  m03.R  m04.R   one method per file: fit function + its cross-validation function
run.R           runs all four methods on the demo data, section by section
```

```r
setwd("<this folder>")
source("run.R", print.eval = TRUE)   # print.eval = TRUE shows the results; plain source() would print nothing
```

Or from a terminal: `Rscript run.R`. In RStudio the script can also be run section by section; the sections build on each other (later tables reuse earlier results), so run them in order.

Runtime of `run.R` on a laptop: logistic under a second, Exact a few seconds, Kernel about a minute, and the network twenty to thirty minutes (it is the only part that needs TensorFlow; Section 8 explains what drives its cost).

The network needs the R package `tensorflow` with a Python TensorFlow 2.15, i.e. Keras 2 with its legacy optimizers (Keras 3 does not have them):

```r
install.packages("tensorflow")
tensorflow::install_tensorflow(version = "2.15")   # needs Python 3.9 to 3.11
```

**Your own data**: replace `demo_data.csv` by a csv with a 0/1 column `Y` and numeric covariate columns, then set at the top of `run.R`

- `x_cols`: the covariate columns (no intercept column; categorical covariates as dummy variables);
- the split: how many cases and controls go to the training set (or use all rows for training and drop the test evaluation);
- `rho`: the prevalence in the target population, not the share of cases in your file;
- `tau`: the PPV lower bound the rule should satisfy;
- and keep both groups reasonably large: every CV fold must contain cases and controls, so each group needs well over `K` rows.
