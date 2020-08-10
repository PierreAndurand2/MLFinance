# MLFinance
We study different ML trading strategies on Futures (examples here are Brent futures (CO1), ...
... Copper futures (HG1), 10-year Treasuries (TY1), Nasdaq Futures (NQ1), SP500 futures (SP500), Gold Futures (GC1))
main_notebook optimizes between Algorithmic Trading Algorithms with fixed parameters having a certain logic to them
Long Term Moving Average: Simple Moving Average(200)
Medium Term Moving Average: SMA(50)
Short Term Moving Average: SMA(20)
MACD(50,200): when medium term crosses long term
MACD(8,20): when very short term crosses short term
Bollinger(20) with 2std
RSI(14)
Other ATS can easily be added
First an optimisation is run in sample on training data to get the optimal weights between ATS for each single asset subject to risk management constaints 
(max total leverage of 2 in absolute terms). The goodness function being maximized is the Sharpe ratio. 
We then have a return stream per asset, and we run a second optimisation to get the weights between assets subject to the sum of weigths being equal to 1
We use a long training period (about 10 years), and then a long testing period (10 years). The goodness function being optimised is the Sharpe ratio.
We made sure that the training period incorporated very different types of markets (trending, range bound, and crisis)

###NOTEBOOKS###
1) main_notebook.ipynb runs the double optimization on a portfolio of assets, and ATS with fixed parameters
2) It calls project_lib2.py
3) main2.py runs it in Python and asks the user for futures contracts to be input, lot size of contracts, amount of money to be managed, start date of training period,...
   ... end date of testing period. And percentage of testing data
   It outputs in sample and out of sample charts on each contract and of the optimised portfolio. It also outputs main metrics per period:
   Sharpe ratio, volatility, total and annualised returns
   It also outputs trades to be done by the close of the day, and historical lot trades per contract for a constant level of AUM
   It also shows to total amount of net leverage across all contracts
4) optimisation_parameters_ATS.ipynb finds which optimal parameters are for each trading strategy on training data per contract. We run a grid search on them
5) main_optimisation_with_optimised_TAS_parameters.ipynb runs the double optimisation with the optimised parameters found in 4)
6) 5) calls project_lib3.py which has its code modified and takes as input the optimised weights found in 4)
7) main_with_LR_TAS_parameters_optimisation.ipynb runs the same program but with parameters optimised on training data which maximize Sharpe ratio on a Linear Regression. 
   It is done with coordinate ascent. We wanted to see if it could bring better results and had some form of transfer learning brought in
8) 7) calls project_lib4.py with the new TAS parameters
9) TraditionalML.ipynb runs AdaBoostClassifier and RandomForetClassifier on the same training data and for each asset. 
   They are then combined assuming we invest in each asset with equal weights. It ouputs the PnL chart per asset and for the portfolio
   as well as the main metrics: Sharpe, returns and volatility. Accuracy on training and testing data as well as confusion matrices are also output per contract
   
###RESULTS###
1) Double optimisation on ATS without optimised parameters bring the best results. Better than Traditional ML methods such as AdaBoost and RandomForest
   Optimising ATS parameters brings overfitting, and has better results on training data, but worse results on testing data

###NOTES###
We can input any portfolio of contracts, and add trading strategies to it easily
