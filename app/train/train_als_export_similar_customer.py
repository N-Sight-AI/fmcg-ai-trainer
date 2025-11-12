"""
Train ALS on implicit purchase data and export:
  - NSR.als_customer_factors_wide (customer_id, f1..fk, l2_norm)
  - (optional) NSR.customer_neighbors_hybrid rows with sim_als only (top M)

Dependencies:
  pip install pandas numpy scipy implicit pyodbc tqdm

Notes:
  - This script treats each (customer, item) with confidence = number of orders
    in the lookback window; you can swap for log(1+count) or binary.
  - Make sure k in this script matches your SQL table column count.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from tqdm import tqdm
from sqlalchemy import create_engine, text

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from app.shared.common.tracing import get_logger, clean_logs_if_enabled

# Setup logging for this script using environment variables
log_level = os.getenv('NSIGHT_LOG_LEVEL', 'INFO')
log_format = os.getenv('NSIGHT_LOG_FORMAT', 'json')

# Custom logging setup for order recommender training
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# Create logs directory in project root (must resolve __file__ first to get absolute path)
log_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "logs"
clean_logs_if_enabled(str(log_dir))
log_dir.mkdir(exist_ok=True)

# Setup custom logger for training
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level.upper()))
logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, log_level.upper()))
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler for order recommender logs
log_file = log_dir / f"training-{datetime.now().strftime('%Y-%m-%d')}.log"
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(getattr(logging, log_level.upper()))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# --------------------------
# Config
# --------------------------
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))
DOC_TYPES     = ("invoice", "return")
K_FACTORS     = int(os.getenv("ALS_K", "32"))       # must match SQL table width (f1..fK)
ALPHA_COO     = float(os.getenv("ALS_CONF_ALPHA", "1.0"))  # confidence scaling if needed
ALS_ITERS     = int(os.getenv("ALS_ITERS", "25"))
ALS_REG       = float(os.getenv("ALS_REG", "0.05"))
ALS_THREADS   = int(os.getenv("ALS_THREADS", "0"))
ALS_USE_CG    = True  # Conjugate Gradient solver tends to be fast/stable

# Optional neighbor export
WRITE_ALS_NEIGHBORS   = True
ALS_TOP_M_PER_CUSTOMER = int(os.getenv("ALS_TOP_M", "100"))  # store top M ALS neighbors/customer

# Database connection settings
CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", 300))
COMMAND_TIMEOUT = int(os.getenv("DB_COMMAND_TIMEOUT", 600))

# --------------------------
# Main Class
# --------------------------
class ALSSimilarCustomerTrainer:
    """ALS model trainer for similar customer export."""
    
    def __init__(self, tenant: str):
        """
        Initialize the ALS trainer.
        
        Args:
            tenant: Tenant identifier
        """
        self.tenant = tenant
        self.logger = logger
        
        # Test logging setup
        self.logger.info(f"ALS Similar Customer Trainer initialized for tenant: {tenant}")
        
        # Debug: Check if logging is working
        import logging
        self.logger.info(f"Logger level: {self.logger.level if hasattr(self.logger, 'level') else 'unknown'}")
        self.logger.info(f"Logger handlers: {len(logging.getLogger().handlers)}")
        
        # Use tenant config for database connection
        self.tenant_config = self._get_tenant_config_from_config(tenant)
        
        # Initialize database engine using unified database manager
        self.engine = self._create_engine()
    
    def _get_tenant_config_from_config(self, tenant: str):
        """Get tenant config from config.json file."""
        try:
            import json
            from app.shared.common.tenant_utils import get_tenant_config_from_cache_only
            
            # Try to get from cache first (if available)
            try:
                tenant_config = get_tenant_config_from_cache_only(tenant)
                if tenant_config:
                    self.logger.info(f"Retrieved tenant config for {tenant} from cache")
                    return tenant_config
            except Exception as e:
                self.logger.warning(f"Failed to get tenant config from cache: {e}")
            
            # Fallback to config.json
            # Create a simple tenant config class since the module doesn't exist
            class TenantConfig:
                def __init__(self, tenant_id, name, description, db_server, db_name, db_driver, 
                           db_user, db_port, db_password, db_trusted_connection, db_encrypt, db_trust_server_cert):
                    self.tenant_id = tenant_id
                    self.name = name
                    self.description = description
                    self.db_server = db_server
                    self.db_name = db_name
                    self.db_driver = db_driver
                    self.db_user = db_user
                    self.db_port = db_port
                    self.db_password = db_password
                    self.db_trusted_connection = db_trusted_connection
                    self.db_encrypt = db_encrypt
                    self.db_trust_server_cert = db_trust_server_cert
            
            # Load config from config.json
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get tenant configuration
            tenant_config_data = config.get('tenants', {}).get(tenant.upper())
            if not tenant_config_data:
                raise ValueError(f"Tenant '{tenant}' not found in config.json")
            
            # Create tenant config from config.json
            tenant_config = TenantConfig(
                tenant_id=tenant,
                name=f"Tenant {tenant}",
                description=f"Tenant {tenant} from config.json",
                db_server=tenant_config_data['db_server'],
                db_name=tenant_config_data['db_name'],
                db_driver=tenant_config_data.get('db_driver', 'ODBC Driver 17 for SQL Server'),
                db_user=tenant_config_data.get('db_user'),
                db_port=tenant_config_data.get('db_port', 1433),
                db_password=tenant_config_data.get('db_password'),
                db_trusted_connection=tenant_config_data.get('db_trusted_connection', False),
                db_encrypt=tenant_config_data.get('db_encrypt', False),
                db_trust_server_cert=tenant_config_data.get('db_trust_server_cert', True)
            )
            
            self.logger.info(f"Created tenant config for {tenant} from config.json")
            return tenant_config
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant config from config.json for {tenant}: {e}")
            raise
    
    def _create_engine(self):
        """Create database engine for the tenant."""
        try:
            if not self.tenant_config:
                raise ValueError(f"Tenant config not available for {self.tenant}")
            
            # Create connection string
            if self.tenant_config.db_trusted_connection:
                connection_string = (
                    f"mssql+pyodbc://{self.tenant_config.db_server}:{self.tenant_config.db_port}/"
                    f"{self.tenant_config.db_name}?driver={self.tenant_config.db_driver}&"
                    f"Trusted_Connection=yes"
                )
            else:
                connection_string = (
                    f"mssql+pyodbc://{self.tenant_config.db_user}:{self.tenant_config.db_password}@"
                    f"{self.tenant_config.db_server}:{self.tenant_config.db_port}/"
                    f"{self.tenant_config.db_name}?driver={self.tenant_config.db_driver}"
                )
            
            # Create engine
            engine = create_engine(
                connection_string,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_recycle=3600,
                echo=False
            )
            
            self.logger.info(f"Database engine created for tenant {self.tenant}")
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to create database engine for tenant {self.tenant}: {str(e)}")
            raise


    def load_txn_counts(self):
        """Load transaction counts from the database."""
        try:
            # Build (customer_id, item_id, count) over lookback window
            sql = f"""
DECLARE @asof_utc DATETIME2(3) = SYSUTCDATETIME();
SELECT
  CAST(t.customer_id AS BIGINT) AS customer_id,
  CAST(tl.item_id     AS BIGINT) AS item_id,
  COUNT(1)                           AS cnt
FROM NSR.transactions t
JOIN NSR.transaction_lines tl ON tl.txn_id = t.txn_id
WHERE t.txn_date >= DATEADD(DAY, -{LOOKBACK_DAYS}, @asof_utc)
  AND t.doc_type IN ({",".join(["'{}'".format(dt) for dt in DOC_TYPES])})
GROUP BY t.customer_id, tl.item_id
"""
            df = pd.read_sql(sql, self.engine)
            self.logger.info(f"Loaded {len(df)} transaction counts for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load transaction counts for tenant {self.tenant}: {str(e)}")
            raise

    def build_mappings(self, df):
        """Build mappings for sparse matrix construction."""
        # Map to 0..U-1 and 0..I-1 for sparse matrix
        cust_ids = df["customer_id"].astype(np.int64).unique()
        item_ids = df["item_id"].astype(np.int64).unique()
        cust_ids.sort()
        item_ids.sort()
        
        self.logger.info(f"Mapped {len(cust_ids)} unique customers and {len(item_ids)} unique items")
        self.logger.info(f"Customer ID range: {cust_ids.min()} to {cust_ids.max()}")
        self.logger.info(f"Item ID range: {item_ids.min()} to {item_ids.max()}")
        
        cust_id2row = {cid: i for i, cid in enumerate(cust_ids)}
        item_id2col = {iid: i for i, iid in enumerate(item_ids)}
        return cust_ids, item_ids, cust_id2row, item_id2col

    def to_csr(self, df, cust_id2row, item_id2col):
        """Convert dataframe to CSR matrix."""
        rows = df["customer_id"].map(cust_id2row).values.astype(np.int32)
        cols = df["item_id"].map(item_id2col).values.astype(np.int32)
        data = df["cnt"].astype(np.float32).values
        # Optional confidence transform
        if ALPHA_COO != 1.0:
            data = (ALPHA_COO * data).astype(np.float32)
        mat = sparse.coo_matrix((data, (rows, cols)),
                                shape=(len(cust_id2row), len(item_id2col))).tocsr()
        return mat

    def train_als(self, user_item_csr):
        """Train ALS model on user-item matrix with Windows BLAS workaround."""
        # implicit ALS expects item-user matrix; we transpose
        item_user = user_item_csr.T.tocsr()
        
        self.logger.info(f"Matrix shape: {item_user.shape}, dtype: {item_user.dtype}")
        self.logger.info(f"Matrix nnz (non-zero elements): {item_user.nnz}")
        
        # Windows BLAS workaround
        import os
        import threadpoolctl
        
        # Set environment variables to prevent BLAS conflicts
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['THREADPOOLCTL_THREAD_LIMIT'] = '1'
        os.environ['IMPLICIT_NUM_THREADS'] = '1'
        
        # Windows tqdm workaround - disable progress bar in subprocess
        import sys
        if hasattr(sys, 'stderr') and hasattr(sys.stderr, 'isatty'):
            if not sys.stderr.isatty():
                os.environ['TQDM_DISABLE'] = '1'
        
        # Store original threadpoolctl function
        original_threadpool_info = threadpoolctl.threadpool_info
        
        try:
            # Apply threadpoolctl workaround
            self.logger.info(f"Applied threadpoolctl workaround for tenant {self.tenant}")
            
            def patched_threadpool_info():
                try:
                    return original_threadpool_info()
                except OSError as e:
                    if "Microsoft Visual C++ 14.0 is required" in str(e):
                        self.logger.warning("Bypassing threadpoolctl check due to Windows BLAS issue")
                        return []
                    else:
                        raise
            
            # Monkey patch threadpoolctl
            threadpoolctl.threadpool_info = patched_threadpool_info
            
            model = AlternatingLeastSquares(
                factors=int(K_FACTORS),
                regularization=float(ALS_REG),
                iterations=int(ALS_ITERS),
                use_cg=bool(ALS_USE_CG),
                num_threads=int(ALS_THREADS) if ALS_THREADS > 0 else 0,
            )
            
            # For implicit data, we typically use confidence = 1 + alpha * r
            # Here we directly pass the matrix; you can also use bm25_weight or tfidf weighting.
            self.logger.info("Starting ALS model fitting...")
            model.fit(item_user, show_progress=False)
            
        finally:
            # Restore original threadpoolctl function
            threadpoolctl.threadpool_info = original_threadpool_info
            self.logger.info(f"Restored threadpoolctl function for tenant {self.tenant}")
        
        # Note: Since we transposed to item_user, model.user_factors corresponds to items
        # and model.item_factors corresponds to users (customers)
        # For similar customers, we want the user factors (which are now in item_factors)
        user_factors = model.item_factors  # shape [U, K] - these are actually user factors
        self.logger.info(f"ALS training completed. User factors shape: {user_factors.shape}")
        return user_factors

    def write_customer_factors(self, cust_ids, user_factors):
        """Write customer factors to database with optimized bulk insert methods."""
        try:
            # Build DataFrame with f1..fK + l2_norm
            norms = np.linalg.norm(user_factors, axis=1)
            self.logger.info(f"Building DataFrame with {len(cust_ids)} customers and {K_FACTORS} factors")
            
            data = {"customer_id": cust_ids, "l2_norm": norms}
            for j in range(K_FACTORS):
                data[f"f{j+1}"] = user_factors[:, j]
            df = pd.DataFrame(data)
            self.logger.info(f"DataFrame created successfully with shape: {df.shape}")

            # Use optimized export method based on dataset size
            if len(df) > 10000000:  # 10M+ records
                # Try BULK INSERT first, fallback to super-fast if it fails
                try:
                    self._export_customer_factors_bulk_insert(df)
                except Exception as e:
                    self.logger.warning(f"BULK INSERT failed, falling back to super-fast method: {str(e)}")
                    self._export_customer_factors_super_fast(df)
            elif len(df) > 100000:  # 100K+ records
                self._export_customer_factors_super_fast(df)
            else:
                self._export_customer_factors_standard(df)
                
            self.logger.info(f"Customer factors written for tenant {self.tenant}")
            
        except Exception as e:
            self.logger.error(f"Failed to write customer factors for tenant {self.tenant}: {str(e)}")
            raise

    def _export_customer_factors_standard(self, df):
        """Standard export method for smaller datasets."""
        with self.engine.connect() as conn:
            # Stage into a temp table (no timestamp columns)
            create_stage = f"""
IF OBJECT_ID('als_stage') IS NOT NULL DROP TABLE als_stage;
CREATE TABLE als_stage(
  customer_id BIGINT NOT NULL,
  l2_norm     FLOAT  NOT NULL,
  {", ".join([f"f{j+1} FLOAT NOT NULL" for j in range(K_FACTORS)])}
);
"""
            conn.execute(text(create_stage))

            # Prepare data for bulk insert (exclude computed_utc to avoid timestamp issues)
            bulk_df = df[["customer_id", "l2_norm"] + [f"f{j+1}" for j in range(K_FACTORS)]].copy()
            
            # Ensure all columns are proper types
            for col in bulk_df.columns:
                if col == 'customer_id':
                    bulk_df[col] = bulk_df[col].astype('int64')
                elif col == 'l2_norm' or col.startswith('f'):
                    bulk_df[col] = bulk_df[col].astype('float64')
            
            # Use pandas to_sql with method='multi' for bulk insert
            max_params_per_chunk = 2000  # Conservative limit
            columns_per_row = 34  # customer_id, l2_norm, f1-f32
            chunk_size = max_params_per_chunk // columns_per_row  # ~58 rows per chunk
            
            for i in tqdm(range(0, len(bulk_df), chunk_size), desc="Uploading factors (standard)"):
                chunk_df = bulk_df.iloc[i:i+chunk_size].copy()
                
                chunk_df.to_sql(
                    name='als_stage', 
                    con=conn, 
                    if_exists='append' if i > 0 else 'replace', 
                    index=False,
                    method='multi',
                    chunksize=1000
                )
                conn.commit()

            # Clear live table and insert from stage
            conn.execute(text("DELETE FROM NSR.als_customer_factors_wide"))
            conn.commit()
            
            # Insert from stage table to live table
            insert_sql = f"""
INSERT INTO NSR.als_customer_factors_wide (
  customer_id, l2_norm, {", ".join([f"f{j+1}" for j in range(K_FACTORS)])}
)
SELECT 
  customer_id, l2_norm, {", ".join([f"f{j+1}" for j in range(K_FACTORS)])}
FROM als_stage
"""
            conn.execute(text(insert_sql))
            conn.commit()

            # Clean up stage table
            conn.execute(text("DROP TABLE IF EXISTS als_stage"))
            conn.commit()

    def _export_customer_factors_super_fast(self, df):
        """Super-fast bulk export optimized for large datasets."""
        try:
            self.logger.info(f"Super-fast customer factors export: {len(df)} records for tenant {self.tenant}")
            
            # For large datasets, use optimized chunking with larger chunks
            super_chunk_size = 50000  # 50K records per chunk for super-fast operations
            successful_chunks = 0
            failed_chunks = 0
            total_chunks = (len(df) + super_chunk_size - 1) // super_chunk_size
            
            # Get connection once and reuse
            conn = self.engine.connect()
            raw_conn = conn.connection.driver_connection
            cursor = raw_conn.cursor()
            
            try:
                # Enable all performance optimizations
                cursor.fast_executemany = True
                
                # Truncate table first
                cursor.execute("DELETE FROM NSR.als_customer_factors_wide")
                
                # Prepare bulk insert SQL
                columns = ["customer_id", "l2_norm"] + [f"f{j+1}" for j in range(K_FACTORS)]
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"""
                    INSERT INTO NSR.als_customer_factors_wide 
                    ({", ".join(columns)}) 
                    VALUES ({placeholders})
                """
                
                # Process data in super-large chunks for maximum efficiency
                for i in range(0, len(df), super_chunk_size):
                    chunk = df.iloc[i:i + super_chunk_size]
                    chunk_num = i // super_chunk_size + 1
                    
                    try:
                        # Convert to list of tuples for maximum speed
                        data = [tuple(row) for row in chunk[columns].values]
                        
                        # Bulk insert with fast_executemany
                        cursor.executemany(insert_sql, data)
                        raw_conn.commit()
                        
                        self.logger.info(f"Super-fast chunk {chunk_num}/{total_chunks} exported successfully ({len(chunk)} rows)")
                        successful_chunks += 1
                        
                    except Exception as chunk_error:
                        self.logger.error(f"Super-fast chunk {chunk_num} failed: {str(chunk_error)}")
                        failed_chunks += 1
                        raise chunk_error
                
                self.logger.info(f"Super-fast export completed: {successful_chunks} successful chunks, {failed_chunks} failed chunks")
                
            finally:
                # Clean up connections
                cursor.close()
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Super-fast customer factors export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to standard export method
            self.logger.info("Falling back to standard export method...")
            self._export_customer_factors_standard(df)

    def _export_customer_factors_bulk_insert(self, df):
        """Ultimate performance: BULK INSERT using temporary CSV file for millions of records."""
        try:
            self.logger.info(f"BULK INSERT customer factors export: {len(df)} records for tenant {self.tenant}")
            
            import tempfile
            import os
            
            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
                temp_filename = temp_file.name
                
                # Write CSV data directly to file
                columns = ["customer_id", "l2_norm"] + [f"f{j+1}" for j in range(K_FACTORS)]
                df[columns].to_csv(
                    temp_file, 
                    index=False, 
                    header=False,
                    sep='|',  # Use pipe separator for better performance
                    float_format='%.6f'
                )
            
            try:
                # Get connection
                conn = self.engine.connect()
                raw_conn = conn.connection.driver_connection
                cursor = raw_conn.cursor()
                
                try:
                    # Truncate table first
                    cursor.execute("DELETE FROM NSR.als_customer_factors_wide")
                    
                    # Use BULK INSERT for maximum performance
                    bulk_insert_sql = f"""
                        BULK INSERT NSR.als_customer_factors_wide
                        FROM '{temp_filename}'
                        WITH (
                            FIELDTERMINATOR = '|',
                            ROWTERMINATOR = '\\n',
                            BATCHSIZE = 100000,
                            TABLOCK,
                            CHECK_CONSTRAINTS
                        )
                    """
                    
                    self.logger.info(f"Executing BULK INSERT from temporary file: {temp_filename}")
                    cursor.execute(bulk_insert_sql)
                    raw_conn.commit()
                    
                    self.logger.info(f"BULK INSERT completed: {len(df)} records inserted successfully")
                    
                finally:
                    cursor.close()
                    conn.close()
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                    self.logger.info("Temporary file cleaned up")
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"BULK INSERT customer factors export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to super-fast export method
            self.logger.info("Falling back to super-fast export method...")
            self._export_customer_factors_super_fast(df)

    def block_candidates(self):
        """Load customer blocking attributes."""
        try:
            # Optional blocking attributes (region/channel/segment).
            # If you don't have NSR.customer_attr, return empty DF to skip blocking in Python.
            if pd.read_sql("SELECT 1 FROM sys.objects WHERE name='customer_attr' AND schema_id=SCHEMA_ID('NSR')", self.engine).empty:
                return pd.DataFrame(columns=["customer_id", "region_id", "channel_id", "segment_id"])
            sql = """
SELECT CAST(customer_id AS BIGINT) AS customer_id,
       region_id, channel_id, segment_id
FROM NSR.customer_attr
"""
            df = pd.read_sql(sql, self.engine)
            self.logger.info(f"Loaded {len(df)} customer blocking attributes for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load customer blocking attributes for tenant {self.tenant}: {str(e)}")
            raise

    def export_als_neighbors(self, cust_ids, user_factors, top_m=600,
                             block_df=None, block_scope="segment",
                             k_window=50, alpha=0.60, lookback_days=365, min_overlap_items=3):
        """
        Compute pure ALS cosine neighbors per customer (top_m),
        write rows into NSR.customer_neighbors_hybrid with:
          sim_overlap=NULL, sim_als=..., sim_final=sim_als
        Your SQL proc will later blend overlap and ALS to produce sim_final.
        """
        try:
            self.logger.info(f"Starting ALS neighbors export for tenant {self.tenant}")
            self.logger.info(f"Parameters: top_m={top_m}, k_window={k_window}, alpha={alpha}, lookback_days={lookback_days}, min_overlap_items={min_overlap_items}")
            self.logger.info(f"Customer factors shape: {user_factors.shape}, Customer IDs count: {len(cust_ids)}")
            with self.engine.connect() as conn:
                conn.execute(text("IF OBJECT_ID('cnh_stage') IS NOT NULL DROP TABLE cnh_stage;"))
                conn.execute(text("""
CREATE TABLE cnh_stage(
  customer_id BIGINT NOT NULL,
  neighbor_id BIGINT NOT NULL,
  sim_overlap FLOAT NULL,
  sim_als     FLOAT NULL,
  sim_final   FLOAT NOT NULL,
  k_window    INT   NOT NULL,
  alpha       FLOAT NOT NULL,
  lookback_days INT NOT NULL,
  min_overlap_items INT NOT NULL
);
"""))
                conn.commit()
                self.logger.info("CNH stage table created successfully")

                # Precompute norms (already computed above, but recompute to be safe)
                norms = np.linalg.norm(user_factors, axis=1)
                norms[norms == 0] = 1e-12
                self.logger.info(f"Precomputed norms for {len(norms)} customers")

                # In-memory ID -> row index
                row_index = {cid: i for i, cid in enumerate(cust_ids)}

                # Optionally build blocks to restrict candidate set
                # dict: block_key -> list of row indices
                block_map = None
                if block_df is not None and not block_df.empty and block_scope in ("region", "channel", "segment"):
                    key_col = {"region": "region_id", "channel": "channel_id", "segment": "segment_id"}[block_scope]
                    merged = pd.DataFrame({"customer_id": cust_ids}).merge(block_df[["customer_id", key_col]], on="customer_id", how="left")
                    block_map = {}
                    for key, grp in merged.groupby(key_col, dropna=True):
                        block_map[key] = grp["customer_id"].map(row_index).tolist()

                # Use optimized batch insertion method
                self._export_neighbors_optimized(
                    conn, cust_ids, user_factors, norms, block_map, block_scope, 
                    block_df, top_m, k_window, alpha, lookback_days, min_overlap_items
                )

                # Clear live table and insert from stage (avoid timestamp column issues)
                self.logger.info("Clearing live table NSR.customer_neighbors_hybrid")
                conn.execute(text("DELETE FROM NSR.customer_neighbors_hybrid"))
                conn.commit()
                
                # Insert from stage table to live table (exclude timestamp columns - let SQL Server handle them)
                self.logger.info("Inserting neighbor data from stage table to live table")
                conn.execute(text("""
INSERT INTO NSR.customer_neighbors_hybrid (
  customer_id, neighbor_id, sim_overlap, sim_als, sim_final,
  k_window, alpha, lookback_days, min_overlap_items
)
SELECT 
  customer_id, neighbor_id, sim_overlap, sim_als, sim_final,
  k_window, alpha, lookback_days, min_overlap_items
FROM cnh_stage
"""))
                conn.commit()
                
                # Clean up stage table
                conn.execute(text("DROP TABLE IF EXISTS cnh_stage"))
                conn.commit()
                
            self.logger.info(f"ALS neighbors exported for tenant {self.tenant}")
            
        except Exception as e:
            self.logger.error(f"Failed to export ALS neighbors for tenant {self.tenant}: {str(e)}")
            raise

    def _export_neighbors_optimized(self, conn, cust_ids, user_factors, norms, block_map, block_scope, 
                                   block_df, top_m, k_window, alpha, lookback_days, min_overlap_items):
        """Optimized neighbors export with multiple performance tiers."""
        try:
            # Determine batch size based on dataset size
            total_customers = len(cust_ids)
            if total_customers > 1000000:  # 1M+ customers
                batch_size = 100000
                self.logger.info(f"Using large dataset batch size: {batch_size}")
            elif total_customers > 100000:  # 100K+ customers
                batch_size = 50000
                self.logger.info(f"Using medium dataset batch size: {batch_size}")
            else:
                batch_size = 20000
                self.logger.info(f"Using standard batch size: {batch_size}")
            
            insert_rows = []
            processed_customers = 0
            
            self.logger.info(f"Starting optimized neighbor computation for {total_customers} customers")

            for idx, cust_id in tqdm(list(enumerate(cust_ids)), desc="Neighbors (ALS)"):
                vec = user_factors[idx]
                if norms[idx] <= 1e-12:
                    continue

                # Candidates
                if block_map is None:
                    cand_indices = np.arange(len(cust_ids), dtype=np.int64)
                else:
                    # pick block for this customer
                    if block_scope == "region":
                        key_col = "region_id"
                    elif block_scope == "channel":
                        key_col = "channel_id"
                    else:
                        key_col = "segment_id"
                    # get key
                    key = None
                    if block_df is not None and not block_df.empty:
                        row = block_df.loc[block_df["customer_id"] == cust_id]
                        if not row.empty:
                            key = row.iloc[0][key_col]
                    cand_indices = np.array(block_map.get(key, []), dtype=np.int64)

                # exclude self
                cand_indices = cand_indices[cand_indices != idx]
                if cand_indices.size == 0:
                    continue

                # cosine = (V dot Ws) / (||V|| * ||W||)
                cand_vecs = user_factors[cand_indices]
                dots = cand_vecs @ vec
                cos = dots / (norms[cand_indices] * norms[idx])

                # top M
                if cos.size > top_m:
                    top_idx = np.argpartition(-cos, top_m-1)[:top_m]
                else:
                    top_idx = np.arange(cos.size)

                for j in top_idx:
                    nb_idx = cand_indices[j]
                    nb_id  = int(cust_ids[nb_idx])
                    s = float(cos[j])
                    insert_rows.append((
                        int(cust_id), nb_id,
                        None,          # sim_overlap
                        s,             # sim_als
                        s,             # sim_final (pure ALS here)
                        int(k_window), float(alpha),
                        int(lookback_days), int(min_overlap_items)
                    ))
                
                processed_customers += 1
                
                # Process batch when size is reached
                if len(insert_rows) >= batch_size:
                    self._flush_neighbors_batch(conn, insert_rows)
                    insert_rows = []
                    self.logger.info(f"Processed {processed_customers}/{total_customers} customers")

            # Flush remaining rows
            if insert_rows:
                self._flush_neighbors_batch(conn, insert_rows)
                self.logger.info(f"Final batch processed. Total customers processed: {processed_customers}")
                
        except Exception as e:
            self.logger.error(f"Optimized neighbors export failed: {str(e)}")
            raise

    def _flush_neighbors_batch(self, conn, insert_rows):
        """Flush a batch of neighbor rows to database with optimized method."""
        try:
            self.logger.info(f"Processing batch of {len(insert_rows)} neighbor rows")
            
            # Convert rows to DataFrame
            cnh_df = pd.DataFrame(insert_rows, columns=[
                'customer_id', 'neighbor_id', 'sim_overlap', 'sim_als', 'sim_final',
                'k_window', 'alpha', 'lookback_days', 'min_overlap_items'
            ])
            
            # Use optimized bulk insert method based on batch size
            if len(insert_rows) > 50000:  # Large batch
                self._bulk_insert_neighbors_fast(conn, cnh_df)
            else:  # Standard batch
                cnh_df.to_sql(
                    name='cnh_stage',
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=222  # Safe chunk size for 9 columns (1998 parameters max)
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to flush neighbors batch: {str(e)}")
            raise

    def _bulk_insert_neighbors_fast(self, conn, cnh_df):
        """Fast bulk insert for large neighbor batches."""
        try:
            # Get raw connection for fast_executemany
            raw_conn = conn.connection.driver_connection
            cursor = raw_conn.cursor()
            
            try:
                cursor.fast_executemany = True
                
                # Prepare bulk insert SQL
                insert_sql = """
                    INSERT INTO cnh_stage 
                    (customer_id, neighbor_id, sim_overlap, sim_als, sim_final,
                     k_window, alpha, lookback_days, min_overlap_items) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Convert to list of tuples for maximum speed
                data = [tuple(row) for row in cnh_df.values]
                
                # Bulk insert with fast_executemany
                cursor.executemany(insert_sql, data)
                raw_conn.commit()
                
                self.logger.info(f"Fast bulk insert completed: {len(data)} neighbor rows")
                
            finally:
                cursor.close()
                
        except Exception as e:
            self.logger.error(f"Fast bulk insert failed: {str(e)}")
            raise

    def train_and_export(self):
        """Main training and export function."""
        try:
            self.logger.info(f"Starting ALS similar customer training for tenant {self.tenant}")
            
            self.logger.info("Loading transactions...")
            df_counts = self.load_txn_counts()
            if df_counts.empty:
                self.logger.warning("No transactions in lookback window. Exiting.")
                return

            self.logger.info("Building mappings...")
            cust_ids, item_ids, cust_id2row, item_id2col = self.build_mappings(df_counts)

            self.logger.info("Building CSR...")
            ui = self.to_csr(df_counts, cust_id2row, item_id2col)

            self.logger.info(f"Training ALS (k={K_FACTORS}, iters={ALS_ITERS}, reg={ALS_REG})...")
            user_factors = self.train_als(ui)

            self.logger.info("Writing customer factors...")
            self.logger.info(f"Customer IDs length: {len(cust_ids)}, User factors shape: {user_factors.shape}")
            self.write_customer_factors(cust_ids, user_factors)

            if WRITE_ALS_NEIGHBORS:
                self.logger.info("Exporting ALS neighbors (pure cosine)...")
                attrs = self.block_candidates()  # may be empty if table missing
                self.export_als_neighbors(
                    cust_ids=cust_ids,
                    user_factors=user_factors,
                    top_m=ALS_TOP_M_PER_CUSTOMER,
                    block_df=attrs if not attrs.empty else None,
                    block_scope="segment",  # region|channel|segment|None
                    k_window=50,
                    alpha=0.60,
                    lookback_days=LOOKBACK_DAYS,
                    min_overlap_items=3
                )

            self.logger.info(f"ALS similar customer training completed for tenant {self.tenant}")
            
        except Exception as e:
            self.logger.error(f"ALS similar customer training failed for tenant {self.tenant}: {str(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            raise


# --------------------------
# Main
# --------------------------
def main():
    """Main entry point for the script."""
    logger.info(f"Script started - Python version: {sys.version}")
    logger.info(f"Script path: {__file__}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ALS model and export similar customers")
    parser.add_argument("--tenant", required=True, help="Tenant identifier")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual execution)")
    
    args = parser.parse_args()
    logger.info(f"Command line arguments parsed: tenant={args.tenant}, dry_run={args.dry_run}")
    
    try:
        if args.dry_run:
            logger.info(f"Dry run mode: Would train ALS similar customer model for tenant {args.tenant}")
            return
        
        # Debug: Check environment variables
        logger.info(f"Environment variables:")
        logger.info(f"  TENANT_{args.tenant.upper()}_DB_SERVER: {os.getenv(f'TENANT_{args.tenant.upper()}_DB_SERVER', 'NOT_SET')}")
        logger.info(f"  TENANT_{args.tenant.upper()}_DB_NAME: {os.getenv(f'TENANT_{args.tenant.upper()}_DB_NAME', 'NOT_SET')}")
        logger.info(f"  TENANT_{args.tenant.upper()}_DB_USER: {os.getenv(f'TENANT_{args.tenant.upper()}_DB_USER', 'NOT_SET')}")
        
        # Create trainer and run
        logger.info(f"Creating trainer for tenant: {args.tenant}")
        trainer = ALSSimilarCustomerTrainer(args.tenant)
        logger.info(f"Starting training pipeline...")
        trainer.train_and_export()
        logger.info(f"Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        import traceback
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
