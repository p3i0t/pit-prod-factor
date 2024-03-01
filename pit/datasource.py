from typing import Protocol, TYPE_CHECKING

from pandas.core.api import DataFrame as DataFrame

if TYPE_CHECKING:
    import pandas as pl


class DataSource(Protocol):
    def collect(self) -> pl.DataFrame:
        ...
        
        
class OfflineDataSource(DataSource):
    """Data source loading offline data from disk.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def collect(self) -> DataFrame:
        return super().collect()
    
    
class OnlineDataSource(DataSource):
    """Data source loading online data from database using polars.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def collect(self) -> DataFrame:
        return super().collect()
    
class OnlineDatareaderDataSource(DataSource):
    """Data source for inference loading data from datareader.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def collect(self) -> DataFrame:
        return super().collect()