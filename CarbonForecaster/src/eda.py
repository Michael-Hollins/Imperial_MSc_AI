import refinitiv.data as rd
import pandas as pd
import re
from load_raw_data import load_all

# Constants
LARGEST_MINING_COMPANIES=['2222.SE', 'XOM', 'CVX', 'SHEL.L', '601857.SS', 'TTEF.PA', 'GAZP.MM', 'COP', 'BP.L', 'ROSN.MM']
INTERVAL='1Y'
START_DATE='2015-01-01'
END_DATE='2024-01-01'

data = load_all(universe=LARGEST_MINING_COMPANIES, interval=INTERVAL, start_date=START_DATE, end_date=END_DATE )

# Let's check out how many zeros we have
print(data.isin([0]).sum(axis=0)/len(data))
