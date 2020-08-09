import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import pdblp   			#for Bloomberg API
import os
import sys

from scipy.optimize import minimize  #For minimizing and finding weights


# global variables
BBDATACACHEPATH = 'bbcache/';      # MUST END IN /
a = os.makedirs(BBDATACACHEPATH, exist_ok=1 )   

# global variables for the optimizer
F = {}    # will hold our backtest data and run data. It gets overwritten everytime we run a new contract
PF = {}   # will hold portfolio PNL  


BBCACHE = {} # mem cache of BB loads, to limit loads

# for output redirection to avoid scipy optimizer output excess  
SYSNORMALSTDOUT = sys.stdout    
    
# contains variables helping with risk management constraints
def myvars():

    v = {}
    #v['maxleverage']  = 1.75 #max leverage per ATS - passed in the constraint function of the optimizer
    v['leveragecutoffthreshold']  = 2. #For any contract other than 10-year treasuries, max |leverage| per asset cannot be more than 2
    v['leveragecutoffthresholdTY1']  = 4. #For 10-year Treasuries, max |leverage| of 4 as it is less volatile
    v['insamplefactor']  = 0.5 #fraction of total data which is training data (from start_date)
    
    return v


# load data from bloomberg
def bbload(ticker, start_date, end_date):
    
    global BBCACHE
    global BBDATACACHEPATH
        
    name = ticker[:3]
    
    CSVcachefilename = BBDATACACHEPATH + ticker +  '.' + start_date + end_date + '.csv'
	
    if ticker in BBCACHE:
        a = BBCACHE[ticker]
        print('USING CACHED')
		
    else:
   
        # try to load CSV first, it is easier than BB and good for those without BB
        try: 
            a = pd.read_csv(CSVcachefilename, index_col = "date" )
            print('Loaded from CSV ' + CSVcachefilename)
        
        # If that fails, load from BB
        except:
            con = pdblp.BCon(debug=False, port=8194, timeout=5000)
            con.start()
            a = con.bdh(ticker, ['PX_LAST'],  start_date, end_date )
            a.columns=['close']

            #save as csv 
            #a.to_csv(CSVcachefilename)
            #print('Loaded from BB and Saved to '+CSVcachefilename)

        #cache
        BBCACHE[ticker] = a
	
	
    # save in global
    global F
    F['ticker'] = ticker   #keep the ticker
    F['name']   = name  #give it a short name without spaces
	 
    return a



#adjusted brent returns for futures rolls 
def adjBrentreturns(start_date,end_date):
    a=pd.read_csv('CORollindex.csv')
	
    a['rollix']=a['rollix'].astype(int)
    co2=bbload('CO2 COMDTY',start_date,end_date)
    co1=bbload('CO1 COMDTY',start_date,end_date)

    ret=pd.DataFrame()
    ret['co1']=np.log(co1.close/co1.close.shift(1))
    ret['co2']=np.log(co2.close/co2.close.shift(1))
    ret.index=co1.index

    ret['adjret']=[ret['co1'][i] if a['rollix'][i]==0 else ret['co2'][i] for i in range(len(ret['co1'])) ]
    return ret


#creating features for each asset
def feature(df,start_date,end_date):
    #object df gets modified (features are appended to df) 
	
    global F
	
    # returns. Can be adjusted later for futures rolls (overwritten)
    df['ret']=np.log(df.close/df.close.shift(1))
	
    #if F['name']=='CO1':
	#    df['ret']=adjBrentreturns(start_date, end_date)
	#    print('******** BRENT USING ADJ RETURNS ******')
	   
	
	
	# more features
    df['ma50']=df.close.rolling(50).mean()
    df['ma20']=df.close.rolling(20).mean()
    df['ma200']=df.close.rolling(200).mean()
    df['ma8']=df.close.rolling(8).mean()
    df['std20'] = df.ret.rolling(20).std() 
    
    df['boll20u'] = df['ma20'] + 1.2*df['std20']*df['ma20'] * np.sqrt(20) 
    df['boll20d'] = df['ma20'] - 1.2*df['std20']*df['ma20'] * np.sqrt(20)
    df['rsi'] = rsi(df.close,14)
    
    #"risk manager:" 20-day historical volatility
    fact1 = df.ret.rolling(20).std()*np.sqrt(252) 
    
    #volatility weighting: we will divide positions by df['std20norm']. If vols pick up we reduce positions and vice versa
    if F['name']=='TY1': #For 10-year Treasuries
        df['std20norm'] = fact1/0.1 
    elif F['name']=='SP1' or F['name']=='NQ1':  #For Sp500 or Nasdaq futures
        df['std20norm'] = fact1/0.2
    else:
        df['std20norm'] = fact1/0.3 #For all else
	
    v = myvars()
    F['d'] = df
    F['oosStart'] = int(len(df.ret)*v['insamplefactor']) #index where out of sample starts
    
    return df
    
#RSI function
def rsi(prices, n=14):
    pricediff=prices-prices.shift(1)
    upmove=pd.Series()
    downmove=pd.Series()
    RS=pd.Series()
    RSI=pd.Series()
    upmove=pd.Series([pricediff[i] if pricediff[i]>0 else 0 for i in range(len(pricediff))])
    downmove=pd.Series([-pricediff[i] if pricediff[i]<0 else 0 for i in range(len(pricediff))])
    RS=upmove.rolling(n).mean()/downmove.rolling(n).mean()
    RSI=100-100/(1+RS)
    RSI.index=prices.index
    return RSI
    
    
#creating signals for each asset    
def signal(df):
    
    s=pd.DataFrame()    
    s['ma50']=-np.array(df.ma50>df.close).astype(int)+np.array(df.ma50<df.close).astype(int)
    s['ma20']=-np.array(df.ma20>df.close).astype(int)+np.array(df.ma20<df.close).astype(int)
    s['ma200']=-np.array(df.ma200>df.close).astype(int)+np.array(df.ma200<df.close).astype(int)
    s['ma50_200']=np.array(df.ma50>df.ma200).astype(int)-np.array(df.ma50<df.ma200).astype(int)
    s['ma8_20']=np.array(df.ma8>df.ma20).astype(int)-np.array(df.ma8<df.ma20).astype(int)
    #s['1d']=np.array(df.close>df.close.shift(1)).astype(int)-np.array(df.close<df.close.shift(1)).astype(int)
    s['Bollinger']=np.array(df.close<df.boll20d).astype(int)-np.array(df.close>df.boll20u).astype(int)
    s['rsi']=np.array(df['rsi']<25).astype(int)-np.array(df['rsi']>75).astype(int)
    
	#add dates if helpful
    s.index=df.index
    
    # vol weight each signal. Not time variant.
    s = pandaColDivideVector(s,df['std20norm'])
	
    global F
    F['s'] = s
    
    return s

# unweighted PNL. This is the signal * returns only, no weights. Will be applied weights.
def pnl0(df,s):
    ret=np.array(df.ret)
    UW=pd.DataFrame()
    for col in s.columns:
        UW[col]=s[col].shift(1)*ret
    return UW


#calculate leverage of each signal and sum (constrained by risk rules)
# Must have an x vector by now
def leverage(w):
 
    global F
 
    #basic signals
    lev = F['s']*w
    F['lev'] = lev	
    
    #augment a bit
    levsum = leveragesum(lev)
    F['levsum'] = levsum
    F['netlev'] = levsum['sum']
    return levsum


# Calculates the sum of signal leverage. The sum is not equal to the sum of the parts, necessarily.
# We want for example to chop the max leverage off to improve the average
def leveragesum(lev):
    
    lev['sum'] = lev.sum(axis=1)
    
    # chop max leverage
    v = myvars()
    if F['name']=='TY1':
        THRES = v['leveragecutoffthresholdTY1']
    else:
        THRES = v['leveragecutoffthreshold']
    ix = lev['sum']>THRES
    lev['sum'][ix] = THRES

    ix = lev['sum']<-THRES
    lev['sum'][ix] = -THRES

    return lev
    
    
    
# a sharpe measure for the optimizer only (using weights and the UW matrix)
def sharpeW(weights, dret):
    n=len(dret)
    sumsignals = (dret*weights).sum(axis=1)
    cret=np.exp(sumsignals.sum())**(252/n)-1
    print(cret)
    std=np.std(sumsignals)*np.sqrt(252)
    print(std)
    return cret/std


# sharpe for return series (the standard)
def	sharpe(logret):
    n = len(logret)
    p=np.exp(logret.sum())**(252/n)-1
    s=np.std(logret)*np.sqrt(252)
    return p/s


# will be used INSIDE the minimizer function, so only gets the X vector (replaced with cutoff to have higher averages)
#def tradeConstraintsFunc(x):
#    v = myvars()

    #Calculate a leverage on the proposed x
#    global F
#    lev = leverage(x)
	
#    return [-np.max(np.array(lev['sum'] ))+v['maxleverage'], np.min(np.array(lev['sum'] ))+v['maxleverage'] ] 
    
# OPTIMIZER LOSS FUNCTION    
def lossFunc(w):
    
    v = myvars()
        
    global F
	
    UW = F['UW']
	
	# define an out of sample period and store it
    n=int( len(UW) * v['insamplefactor'] )
    F['oosStart'] = n
	
    INSA = UW[0:n]  #CRITICAL
	
	# calculate some interesting quantities for use
    pIS = (INSA * w).sum(axis=1)   #PNL in sample
	 
	# Choose an optimization target 
    optTarget = F['optTarget']
    if optTarget=='sharpe':
        out = -sharpeW(w, INSA)

    elif optTarget=='pnl':
	     out = -sum(pIS)
		 
    elif optTarget=='dd':
         out=maxdrawdown(pIS)

    elif optTarget=='calmar':
         out=-np.exp(sum(pIS))/maxdrawdown(pIS)
	
    else:
	    out = -sharpeW(w, INSA)
	 
    return out



# the core backtesting function. Produces some helpful plots. Per asset. (asset info is overwritten in F)
def backtest():

    global F

    # calculate the unweighted pnl
    F['UW'] = pnl0(F['d'],F['s'])

    #random init of weights. Set bounds.
    n = len(F['s'].columns)
    w0 = n * [ 1/n ]
    BNDS = ((-1,1),)*n
	
    #cons = ({'type': 'ineq','fun': tradeConstraintsFunc })
	
    print ('** Minimize: Target:'+F['optTarget'])    
    	
    #x = w0  
    #res = minimize(lossFunc, w0, tol=1e-6, bounds=BNDS, constraints=cons) #minimize chooses the method between BFGS, L-BFGS-B, and SLSQP
    #res = minimize(lossFunc, w0, method='SLSQP',tol=1e-6, bounds=BNDS) #method=SLSQP -- no more constraints
    
    
    nulloutput()  # stop output to stdout for the min function
    
    res = minimize(lossFunc, w0, method='SLSQP', tol=1e-6, bounds=BNDS, options={'disp': False, 'maxiter': 1e5 } ) #minimize with method SLSQP 
    
    normaloutput()
    
    
    x = res.x
    
	# Now store some calculated quantities for portfolio creation and analysis etc.
	
	#x - store it safely
    F['x'] = x #weigths between ATS for a given asset
    #F['optimres'] = res
		
	#calculate some final output results of the optim vector x
    levsum = leverage(F['x'])
    F['levsum'] = levsum

    #PNL
    F['cumpnl'] = ((F['UW']*F['x']).sum(axis=1).cumsum()).apply(np.exp);	#Cumulative is real PNL (path of 1$)
    F['pnl'] = (F['UW']*F['x']).sum(axis=1)  #LOG PNL	
	
	#some output and plots to help
    print('optimized X:')
    print(F['x'])

 
# plot some key results
def plotresult():
    
	global F
	
	# LEVERAGE
	levsum = F['levsum']
	plt.plot(levsum['sum'])
	plt.title('Leverage for ' + F['ticker'])
	print('Max Leverage:')
	print(np.max(levsum['sum']))
	plt.show();
	
   
	# PNL
	pnl = F['cumpnl']
	n = F['oosStart']
	plt.plot(pnl) #plotting pnl (path of $1)
	plt.scatter(pnl.index[n],pnl[n],color='r') #red point where out of sample starts
	plt.title('PNL ' + F['ticker'])
	plt.show()
	 
    
	 
    
	  
     

# ease of use function to apply vector to each column
def pandaColDivideVector(p, v):
    newpd = pd.DataFrame()
    for col in p.columns:
        newpd[col]=p[col]/v
    return newpd




def yyyymmdd():
    return datetime.now().strftime('%Y%m%d') 
	
	
	
# very simple and fast max drawdown function. Only does the basics for speed!
# r is a log return vector. Probably need some formatting later of structures.
def maxdrawdown (r):
 
    n = len(r)

    # calculate vector of cum returns. DOES NOT WORK FOR REAL RETURNS. so has to be log.
    cr = np.nancumsum(r);

    #preallocate size of dd = size r
    dd  = np.empty(n);

    # calculate drawdown vector
    mx = 0;
    for i in range(n):
        if cr[i] > mx:
             mx = cr[i] 
			 
        dd[i] = mx - cr[i]

    # calculate maximum drawdown 
    DD = max(dd);

    return DD
		
	
	

# OPTIMIZER Portfolio (2nd optimization)
#for a dataframe input of PNL streams, produce the optimal blend vector x (weights between assets)
def pfopt(df):
  
  global PF
  PF = df;
  
  n = len(df.columns)
  w0 = n * [ 1/n ]
  BNDS = ((-1,1),)*n
  cons = ({'type': 'eq','fun': pfConsFunc })
  res = minimize(pfGoodFunc, w0, tol=1e-6, bounds=BNDS, constraints=cons, options={ 'disp': False,  'maxiter': 1e5 } )
  
  return res.x  
		

# Sum of components less than 1
def pfConsFunc(x):
    
	c1 = np.array(-np.sum(x) + 1)   
	return c1
	#return np.concatenate([c1,c2])


# minimize variance?
def pfGoodFunc(x):
   global PF
   
   p = np.sum(PF * x,axis=1) # pnl
   
   v = myvars()
   n = int(len(p) * v['insamplefactor'])
   INSA = p[0:n]
   
   g = - sharpe(INSA) #negative since we minimize
   
   return g
   
# STD out redirect 
def nulloutput():
    f = open(os.devnull, 'w')
    sys.stdout = f
    
# STD out set back to normal. Needs global var    
def normaloutput():
    global SYSNORMALSTDOUT
    sys.stdout = SYSNORMALSTDOUT