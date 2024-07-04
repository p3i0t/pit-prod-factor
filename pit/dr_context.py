import importlib

class DatareaderContext:
    def __init__(self) -> None:
        ...
    
    def __enter__(self):
        try:
            import datareader as dr
        except ImportError:
            raise ImportError("Error: module datareader not found")
        
        dr.CONFIG.database_url["DB72"] = "clickhouse://alpha_read:32729afc@10.25.1.72:9000"
        importlib.reload(dr.URL)
        dr.URL.DB73 = "clickhouse://test_wyw_allread:3794b0c0@10.25.1.73:9000"
        return dr
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        ...