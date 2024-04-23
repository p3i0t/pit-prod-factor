from data_source import select_cols

import polars as pl

def test_select_cols():
    df = pl.DataFrame({'a': [1., 2., 3., 4.], 'b': ['a', 'b', 'c', 'd']})
    
    df.pipe(select_cols, ['a'])    
    
 
   