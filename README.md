# Summary

1. Article
2. Code and instructions
3. Data
4. References
5. Author

# 1. Article

"Evaluating density forecasts" writted by Francis X. Diebold, Todd A. Ghunter and Anthony S. Tay (1997)

International Economic Review volume 39

# 2. Code and instructions
					
You can find the code in the file "Code"

All the implementation is done with Python 3.7.1 in Jupyter Notebook format (.ipynb)

Everything is commented and choices are justified

List of coding files

- Article review.ipynb: it's the main. We discuss about all concept in the article.

- ArmaGarch.ipynb: Implementation of a class ARMA(pm,qm)-Garch(pv,qv) in the aim to estimate a model based on a data.
(- ArmaGarch.py: To allowed "import" for other Notebook)

- Financial data.ipynb: Application of the theorical result on real financial datas. First on S&P daily returns then Euro Stoxx 50.

- Improving density.ipynb: First extension of the article

- Multi-dimensional.ipynb: Second extension of the article

# 3. Data

1. S&P500.txt: daily returns of S&P 500 from 1962 to 1995
2. EUROXSTOXX50.txt daily returns of Euro Stoxx 50 from 1992 to today (2021)


# 4. References

David Ruppert, David S. Matteson. Statistics and Data Analysis for Financial Engineering. Seconde édition, Springer 2015.

Bollerslev Tim, Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, volume 31, 1986.

Silverman B.W. Density Estimation for Statistics and Data Analysis, 1986.

ohn Nelder, Roger Mead. A simplex method for function minimization. Computer Journal volume 7, 1965.

# 5. Author

Antoine Lepeltier

Last update 24/08/2021
