#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALS Training and Export Script for Customer Order Recommendations.

This script trains an ALS (Alternating Least Squares) model on customer-item interactions
and exports the results to the database. It supports multi-tenant configuration and
can be run as a standalone script or called from the scheduler.

Usage:
    python customer_order_recommendations_als.py --tenant <tenant_id>
    python customer_order_recommendations_als.py --help
"""

import argparse
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from sqlalchemy import create_engine, text

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.shared.common.tracing import get_logger
from app.shared.common.tenant_utils import get_tenant_config_from_cache_only

# Setup logging
logger = get_logger(__name__)


class ALSTrainer:
    """ALS model trainer and exporter for customer order recommendations."""
    
    def __init__(self, tenant: str):
        """
        Initialize the ALS trainer.
        
        Args:
            tenant: Tenant identifier
        """
        self.tenant = tenant
        self.logger = logger
        
        # Get tenant configuration
        self.tenant_config = get_tenant_config_from_cache_only(tenant)
        if not self.tenant_config:
            raise ValueError(f"Tenant '{tenant}' not found in cache")
        
        # ALS model parameters
        self.factors = int(os.getenv("ALS_FACTORS", 64))
        self.reg = float(os.getenv("ALS_REG", 0.08))
        self.iters = int(os.getenv("ALS_ITERS", 15))
        self.topk = int(os.getenv("ALS_TOPK", 200))
        self.alpha = float(os.getenv("ALS_ALPHA", 40.0))
        
        # Business logic parameters
        self.due_threshold = float(os.getenv("DUE_THRESHOLD", 0.8))
        self.default_cycle_days = int(os.getenv("DEFAULT_CYCLE_DAYS", 30))
        
        # Export settings
        self.export_filter_oos = os.getenv("EXPORT_FILTER_OOS", "false").strip().lower() in {"1", "true", "yes"}
        self.warehouse_id = os.getenv("WAREHOUSE_ID", "").strip() or None
        self.min_avail_qty = int(os.getenv("MIN_AVAIL_QTY", 1))
        self.safety_frac = float(os.getenv("SAFETY_FRAC", 0.10))
        self.backfill_floor = float(os.getenv("ALS_BACKFILL_FLOOR", 0.05))
        
        # Database settings
        self.connection_timeout = int(os.getenv("DB_CONNECTION_TIMEOUT", 300))
        self.command_timeout = int(os.getenv("DB_COMMAND_TIMEOUT", 600))
        self.export_timeout = int(os.getenv("EXPORT_TIMEOUT", 600))
        self.chunk_size = int(os.getenv("EXPORT_CHUNK_SIZE", 10000))
        self.max_retries = int(os.getenv("EXPORT_MAX_RETRIES", 3))
        
        # Initialize database engine
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create database engine for the tenant."""
        try:
            # Build connection string from tenant config
            if self.tenant_config.db_trusted_connection:
                conn_str = f"mssql+pyodbc://{self.tenant_config.db_server}:{self.tenant_config.db_port}/{self.tenant_config.db_name}?driver={self.tenant_config.db_driver}&trusted_connection=yes"
            else:
                conn_str = f"mssql+pyodbc://{self.tenant_config.db_user}:{self.tenant_config.db_password}@{self.tenant_config.db_server}:{self.tenant_config.db_port}/{self.tenant_config.db_name}?driver={self.tenant_config.db_driver}"
            
            engine = create_engine(
                conn_str,
                pool_timeout=self.connection_timeout,
                pool_recycle=3600,
                echo=False
            )
            
            self.logger.info(f"Database engine created for tenant {self.tenant}")
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to create database engine for tenant {self.tenant}: {str(e)}")
            raise
    
    def load_interactions(self) -> pd.DataFrame:
        """Load customer-item interactions from the database."""
        try:
            df = pd.read_sql(
                """
                SELECT customer_id, item_id, weight
                FROM NSR.vw_interactions
                WHERE customer_id IS NOT NULL
                  AND item_id IS NOT NULL
                  AND weight IS NOT NULL
                  AND TRY_CAST(customer_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(item_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(weight AS FLOAT) IS NOT NULL
                  AND weight > 0
                """, 
                self.engine
            )
            
            df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
            df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").astype("float32")
            df = df.dropna().astype({"customer_id": "int64", "item_id": "int64"})
            
            if df.empty:
                raise RuntimeError("vw_interactions returned 0 usable rows.")
            
            # Confidence scaling for implicit feedback
            if self.alpha > 0:
                df["weight"] = np.log1p(self.alpha * df["weight"]).astype("float32")
            
            self.logger.info(f"Loaded {len(df)} interactions for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load interactions for tenant {self.tenant}: {str(e)}")
            raise
    
    def load_policy(self) -> pd.DataFrame:
        """Load customer policy data."""
        try:
            # Check if allow_repeats column exists
            try:
                df = pd.read_sql(
                    "SELECT customer_id, allow_repeats FROM NSR.customer_segments_dyn", 
                    self.engine
                ).astype({"customer_id": "int64", "allow_repeats": "int8"})
                self.logger.info("Loaded customer policy with allow_repeats column")
            except Exception:
                self.logger.warning("allow_repeats column not found, using default policy")
                customer_ids = pd.read_sql("SELECT DISTINCT customer_id FROM NSR.customer_segments_dyn", self.engine)
                df = customer_ids.copy()
                df['allow_repeats'] = 1
                df = df.astype({"customer_id": "int64", "allow_repeats": "int8"})
            
            self.logger.info(f"Loaded {len(df)} customer policies for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load customer policies for tenant {self.tenant}: {str(e)}")
            raise
    
    def load_reorder_stats(self) -> pd.DataFrame:
        """Load customer-item reorder statistics."""
        try:
            df = pd.read_sql(
                """
                SELECT customer_id, item_id, days_since_last, med_cycle_days, avg_cycle_days
                FROM NSR.customer_item_reorder_stats
                """, 
                self.engine
            ).astype({"customer_id": "int64", "item_id": "int64"})
            
            self.logger.info(f"Loaded {len(df)} reorder stats for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load reorder stats for tenant {self.tenant}: {str(e)}")
            raise
    
    def load_inventory(self) -> pd.DataFrame:
        """Load inventory data."""
        try:
            if not self.export_filter_oos:
                return pd.DataFrame(columns=["item_id", "avail_qty"])
            
            where_clause = "" if self.warehouse_id is None else f"WHERE warehouse_id = {int(self.warehouse_id)}"
            
            df = pd.read_sql(f"""
                WITH inv_raw AS (
                  SELECT CAST(item_id AS BIGINT) AS item_id,
                         CAST(on_hand AS FLOAT) AS on_hand,
                         CAST(allocated AS FLOAT) AS allocated
                  FROM NSR.inventory_onhand
                  {where_clause}
                ),
                inv AS (
                  SELECT item_id,
                         SUM(CASE WHEN on_hand - allocated > 0
                                  THEN on_hand - allocated ELSE 0 END) AS gross_avail
                  FROM inv_raw GROUP BY item_id
                )
                SELECT item_id,
                       CAST(ROUND(COALESCE(gross_avail,0) * (1.0 - {self.safety_frac}),0) AS INT) AS avail_qty
                FROM inv
            """, self.engine).astype({"item_id": "int64", "avail_qty": "int64"})
            
            self.logger.info(f"Loaded {len(df)} inventory records for tenant {self.tenant}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load inventory for tenant {self.tenant}: {str(e)}")
            raise
    
    def build_matrix(self, df: pd.DataFrame):
        """Build interaction matrix from dataframe."""
        try:
            u_codes, u_uniques = pd.factorize(df["customer_id"].values, sort=True)
            i_codes, i_uniques = pd.factorize(df["item_id"].values, sort=True)
            
            X = coo_matrix(
                (df["weight"].values.astype(np.float32), (u_codes.astype(np.int32), i_codes.astype(np.int32))),
                shape=(len(u_uniques), len(i_uniques))
            ).tocsr()
            
            # BM25 weighting often improves quality on implicit counts
            X = bm25_weight(X, K1=100, B=0.8).tocsr()
            
            self.logger.info(f"Built interaction matrix: {X.shape[0]} users, {X.shape[1]} items for tenant {self.tenant}")
            return X, u_uniques, i_uniques
            
        except Exception as e:
            self.logger.error(f"Failed to build matrix for tenant {self.tenant}: {str(e)}")
            raise
    
    def fit_als_autoorient(self, X, use_gpu=False):
        """Fit ALS model with automatic orientation detection."""
        try:
            # Comprehensive workaround for Windows BLAS issue with implicit library
            import os
            import sys
            
            # Set all thread limits to 1 to prevent BLAS conflicts
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
            
            # Monkey patch threadpoolctl to bypass BLAS check on Windows
            original_threadpool_info = None
            try:
                import threadpoolctl
                original_threadpool_info = threadpoolctl.threadpool_info
                
                def safe_threadpool_info():
                    try:
                        return original_threadpool_info()
                    except OSError as e:
                        if "WinError -1066598273" in str(e) or "Windows Error 0xc06d007f" in str(e):
                            self.logger.warning(f"Bypassing threadpoolctl BLAS check due to Windows error for tenant {self.tenant}")
                            return []
                        raise
                
                threadpoolctl.threadpool_info = safe_threadpool_info
                self.logger.info(f"Applied threadpoolctl workaround for tenant {self.tenant}")
            except ImportError:
                pass
            
            try:
                # Canonical orientation for implicit: items × users
                item_user = X.T.tocsr().astype(np.float32)
                
                # Create and fit the model
                model = AlternatingLeastSquares(
                    factors=int(self.factors), 
                    regularization=float(self.reg), 
                    iterations=int(self.iters), 
                    use_gpu=bool(use_gpu)
                )
                model.fit(item_user, show_progress=False)
            finally:
                # Restore original threadpoolctl function
                if original_threadpool_info is not None:
                    try:
                        import threadpoolctl
                        threadpoolctl.threadpool_info = original_threadpool_info
                        self.logger.info(f"Restored threadpoolctl function for tenant {self.tenant}")
                    except ImportError:
                        pass
            
            n_users, n_items = X.shape
            if model.item_factors.shape[0] == n_items:
                self.logger.info(f"ALS model fitted successfully for tenant {self.tenant}")
                return model
            else:
                # Retry with correct orientation: users × items
                self.logger.info(f"Retrying with correct orientation for tenant {self.tenant}")
                user_item = X.tocsr().astype(np.float32)
                model2 = AlternatingLeastSquares(
                    factors=int(self.factors), 
                    regularization=float(self.reg), 
                    iterations=int(self.iters), 
                    use_gpu=bool(use_gpu)
                )
                model2.fit(user_item, show_progress=True)
                
                if model2.item_factors.shape[0] == n_items:
                    self.logger.info(f"ALS model fitted successfully (retry) for tenant {self.tenant}")
                    return model2
                else:
                    raise ValueError(f"ALS model orientation mismatch: expected {n_items} items, got {model2.item_factors.shape[0]}")
            
        except Exception as e:
            self.logger.error(f"Failed to fit ALS model for tenant {self.tenant}: {str(e)}")
            raise
    
    def l2_normalize(self, mat: np.ndarray) -> np.ndarray:
        """L2 normalize matrix rows."""
        nrm = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        return mat / nrm
    
    def is_due(self, row) -> bool:
        """Check if an item is due for reorder."""
        dsl = row.get("days_since_last")
        med = row.get("med_cycle_days")
        avg = row.get("avg_cycle_days")
        
        if pd.isna(dsl):
            return False
        if not pd.isna(med):
            return dsl >= self.due_threshold * med
        if not pd.isna(avg):
            return dsl >= self.due_threshold * avg
        return dsl >= self.default_cycle_days
    
    def per_user_norm_excluding_backfill(self, df: pd.DataFrame, lo=5, hi=95, min_items=15, 
                                        min_span=1e-4, rank_blend=0.30) -> pd.DataFrame:
        """Normalize ALS scores per user with robust min-max scaling."""
        df = df.copy()
        df["als_score_mm"] = 0.0
        
        # Handle backfill recommendations first
        backfill_mask = df["source"] == "backfill"
        df.loc[backfill_mask, "als_score_mm"] = self.backfill_floor
        
        # Handle ALS recommendations
        als_mask = df["source"] == "als"
        if not als_mask.any():
            return df
        
        # Calculate group sizes for each customer
        customer_counts = df[als_mask]["customer_id"].value_counts()
        
        # Handle customers with few items (rank-based normalization)
        customers_with_few_items = customer_counts[customer_counts < min_items].index
        few_items_mask = als_mask & df["customer_id"].isin(customers_with_few_items)
        if few_items_mask.any():
            df.loc[few_items_mask, "als_score_mm"] = 1.0 - (df.loc[few_items_mask, "als_score_rank"] - 1) / (customer_counts[df.loc[few_items_mask, "customer_id"]] - 1).values
        
        # Handle customers with sufficient items (robust min-max normalization)
        sufficient_items_mask = als_mask & ~df["customer_id"].isin(customers_with_few_items)
        if sufficient_items_mask.any():
            # Calculate quantiles per customer
            quantile_df = df[sufficient_items_mask].groupby("customer_id")["als_score_raw"].quantile([lo/100, hi/100]).unstack()
            quantile_df.columns = ['lo_val', 'hi_val']
            quantile_df['span'] = quantile_df['hi_val'] - quantile_df['lo_val']
            
            # Merge quantiles back to main dataframe
            df = df.merge(quantile_df, left_on="customer_id", right_index=True, how="left")
            
            # Determine normalization strategy
            span_too_small_mask = sufficient_items_mask & (df["span"] < min_span)
            minmax_mask = sufficient_items_mask & (df["span"] >= min_span)
            
            # Handle span too small (rank-based normalization)
            if span_too_small_mask.any():
                df.loc[span_too_small_mask, "als_score_mm"] = 1.0 - (df.loc[span_too_small_mask, "als_score_rank"] - 1) / (customer_counts[df.loc[span_too_small_mask, "customer_id"]] - 1).values
            
            # Handle min-max normalization with rank blending
            if minmax_mask.any():
                minmax_scores = np.clip((df.loc[minmax_mask, "als_score_raw"] - df.loc[minmax_mask, "lo_val"]) / df.loc[minmax_mask, "span"], 0, 1)
                rank_scores = 1.0 - (df.loc[minmax_mask, "als_score_rank"] - 1) / (customer_counts[df.loc[minmax_mask, "customer_id"]] - 1).values
                df.loc[minmax_mask, "als_score_mm"] = (1 - rank_blend) * minmax_scores + rank_blend * rank_scores
            
            # Clean up temporary columns
            df = df.drop(columns=['lo_val', 'hi_val', 'span'], errors='ignore')
        
        return df
    
    def train_and_export(self):
        """Main training and export function."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting ALS training and export for tenant {self.tenant}")
            self.logger.info(f"ALS Parameters: factors={self.factors}, reg={self.reg}, iters={self.iters}, topk={self.topk}, alpha={self.alpha}")
            
            # Load data
            self.logger.info("Step 1: Loading interaction data...")
            interactions_df = self.load_interactions()
            
            self.logger.info("Step 2: Loading policy data...")
            policy_df = self.load_policy()
            
            self.logger.info("Step 3: Loading reorder stats...")
            reorder_stats_df = self.load_reorder_stats()
            
            self.logger.info("Step 4: Loading inventory data...")
            inventory_df = self.load_inventory()
            
            # Build interaction matrix
            self.logger.info("Step 5: Building interaction matrix...")
            X, u_uniques, i_uniques = self.build_matrix(interactions_df)
            
            # Fit ALS model
            self.logger.info("Step 6: Training ALS model...")
            model = self.fit_als_autoorient(X)
            
            # Generate recommendations
            self.logger.info("Step 7: Generating recommendations...")
            
            # Get user factors and normalize
            user_factors = self.l2_normalize(model.user_factors)
            item_factors = self.l2_normalize(model.item_factors)
            
            # Compute recommendations
            rec_scores = user_factors @ item_factors.T
            
            # Create recommendations dataframe using vectorized operations
            self.logger.info(f"Creating recommendations for {len(u_uniques)} users with top-{self.topk} items each...")
            
            # Vectorized approach: get top-k indices for all users at once
            # Ensure topk doesn't exceed the number of available items
            actual_topk = min(self.topk, rec_scores.shape[1])
            if actual_topk < self.topk:
                self.logger.warning(f"Reducing topk from {self.topk} to {actual_topk} due to insufficient items for tenant {self.tenant}")
            
            top_k_indices = np.argpartition(-rec_scores, actual_topk-1, axis=1)[:, :actual_topk]
            
            # Sort only the top-k items for each user
            user_indices = np.arange(len(u_uniques))[:, np.newaxis]
            sorted_indices = np.argsort(-rec_scores[user_indices, top_k_indices], axis=1)
            top_k_indices = top_k_indices[user_indices, sorted_indices]
            
            # Extract scores for top-k items
            top_k_scores = rec_scores[user_indices, top_k_indices]
            
            # Create arrays for all recommendations at once
            num_recs = len(u_uniques) * actual_topk
            customer_ids = np.repeat(u_uniques, actual_topk)
            item_ids = i_uniques[top_k_indices.flatten()]
            scores = top_k_scores.flatten()
            ranks = np.tile(np.arange(1, actual_topk + 1), len(u_uniques))
            
            # Create DataFrame directly from arrays
            rec_df = pd.DataFrame({
                "customer_id": customer_ids,
                "item_id": item_ids,
                "als_score_raw": scores.astype(float),
                "als_score_rank": ranks,
                "source": "als"
            })
            
            self.logger.info(f"Generated {len(rec_df)} ALS recommendations")
            
            # Add backfill recommendations for customers with few ALS recommendations
            self.logger.info("Step 8: Adding backfill recommendations...")
            customer_counts = rec_df.groupby("customer_id").size()
            customers_needing_backfill = customer_counts[customer_counts < 10].index
            
            if len(customers_needing_backfill) > 0:
                self.logger.info(f"Adding backfill recommendations for {len(customers_needing_backfill)} customers")
                
                # Get items that are due for reorder
                due_items = reorder_stats_df[reorder_stats_df.apply(self.is_due, axis=1)]
                
                # Vectorized backfill approach
                backfill_recs = []
                for customer_id in customers_needing_backfill:
                    customer_due_items = due_items[due_items["customer_id"] == customer_id]
                    if len(customer_due_items) > 0:
                        top_items = customer_due_items.head(10)
                        backfill_recs.append(pd.DataFrame({
                            "customer_id": customer_id,
                            "item_id": top_items["item_id"].values,
                            "als_score_raw": 0.0,
                            "als_score_rank": 999,
                            "source": "backfill"
                        }))
                
                if backfill_recs:
                    backfill_df = pd.concat(backfill_recs, ignore_index=True)
                    rec_df = pd.concat([rec_df, backfill_df], ignore_index=True)
                    self.logger.info(f"Added {len(backfill_df)} backfill recommendations")
            
            # Normalize scores per user
            self.logger.info("Step 9: Normalizing scores per user...")
            rec_df = self.per_user_norm_excluding_backfill(rec_df)
            
            # Filter by inventory if needed
            if self.export_filter_oos and not inventory_df.empty:
                self.logger.info("Step 10: Filtering by inventory...")
                rec_df_before = len(rec_df)
                rec_df = rec_df.merge(inventory_df, on="item_id", how="left")
                rec_df["avail_qty"] = rec_df["avail_qty"].fillna(0)
                rec_df = rec_df[rec_df["avail_qty"] >= self.min_avail_qty]
                rec_df_after = len(rec_df)
                self.logger.info(f"Inventory filtering: {rec_df_before} -> {rec_df_after} recommendations")
            
            # Export to database
            self.logger.info("Step 11: Exporting to database...")
            # Use appropriate export method based on dataset size
            if len(rec_df) > 10000000:  # 10M+ records
                # Try BULK INSERT first, fallback to super-fast if it fails
                try:
                    self._export_to_database_bulk_insert(rec_df)
                except Exception as e:
                    self.logger.warning(f"BULK INSERT failed, falling back to super-fast method: {str(e)}")
                    self._export_to_database_super_fast(rec_df)
            elif len(rec_df) > 100000:  # 100K+ records
                self._export_to_database_ultra_fast(rec_df)
            else:
                self._export_to_database(rec_df)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"ALS training and export completed for tenant {self.tenant} in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"ALS training and export failed for tenant {self.tenant} after {execution_time:.3f}s: {str(e)}")
            self.logger.error(f"Error traceback:\n{traceback.format_exc()}")
            raise
    
    def _export_to_database(self, rec_df: pd.DataFrame):
        """Export recommendations to database using ultra-fast bulk operations."""
        try:
            self.logger.info(f"Exporting {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            # Use larger chunk size for bulk operations
            bulk_chunk_size = min(50000, len(rec_df))  # Larger chunks for bulk operations
            successful_chunks = 0
            failed_chunks = 0
            total_chunks = (len(rec_df) + bulk_chunk_size - 1) // bulk_chunk_size
            
            # Get connection once and reuse
            conn = self.engine.connect()
            raw_conn = conn.connection.driver_connection
            cursor = raw_conn.cursor()
            
            try:
                # Enable fast_executemany and other performance optimizations
                cursor.fast_executemany = True
                
                # Truncate table first
                cursor.execute("TRUNCATE TABLE NSR.user_rec_als")
                
                # Prepare bulk insert SQL
                insert_sql = """
                    INSERT INTO NSR.user_rec_als 
                    (customer_id, item_id, als_score_raw, als_score_mm, als_score_rank, source) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Process data in large chunks
                for i in range(0, len(rec_df), bulk_chunk_size):
                    chunk = rec_df.iloc[i:i + bulk_chunk_size]
                    chunk_num = i // bulk_chunk_size + 1
                    
                    for attempt in range(self.max_retries):
                        try:
                            # Prepare data for bulk insert - convert to list of tuples for maximum speed
                            data = [tuple(row) for row in chunk[["customer_id", "item_id", "als_score_raw", "als_score_mm", "als_score_rank", "source"]].values]
                            
                            # Bulk insert with fast_executemany
                            cursor.executemany(insert_sql, data)
                            raw_conn.commit()
                            
                            self.logger.info(f"Chunk {chunk_num}/{total_chunks} exported successfully ({len(chunk)} rows)")
                            successful_chunks += 1
                            break
                        
                        except Exception as chunk_error:
                            if attempt < self.max_retries - 1:
                                self.logger.warning(f"Chunk {chunk_num} failed (attempt {attempt + 1}/{self.max_retries}): {str(chunk_error)}")
                                import time
                                time.sleep(2)  # Shorter retry delay for bulk operations
                            else:
                                self.logger.error(f"Chunk {chunk_num} failed after {self.max_retries} attempts")
                                failed_chunks += 1
                                raise chunk_error
                
                self.logger.info(f"Export completed: {successful_chunks} successful chunks, {failed_chunks} failed chunks")
                
            finally:
                # Clean up connections
                cursor.close()
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to export recommendations for tenant {self.tenant}: {str(e)}")
            raise

    def _export_to_database_ultra_fast(self, rec_df: pd.DataFrame):
        """Ultra-fast bulk export using SQLAlchemy bulk operations for very large datasets."""
        try:
            self.logger.info(f"Ultra-fast export: {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            # Use SQLAlchemy bulk operations for maximum speed
            from sqlalchemy import text
            
            # Get connection
            conn = self.engine.connect()
            raw_conn = conn.connection.driver_connection
            cursor = raw_conn.cursor()
            
            try:
                # Enable all performance optimizations
                cursor.fast_executemany = True
                
                # Truncate table
                cursor.execute("TRUNCATE TABLE NSR.user_rec_als")
                
                # Prepare bulk insert SQL
                insert_sql = """
                    INSERT INTO NSR.user_rec_als 
                    (customer_id, item_id, als_score_raw, als_score_mm, als_score_rank, source) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Convert DataFrame to list of tuples for maximum speed
                data = [tuple(row) for row in rec_df[["customer_id", "item_id", "als_score_raw", "als_score_mm", "als_score_rank", "source"]].values]
                
                # Single bulk insert for maximum speed
                self.logger.info(f"Performing single bulk insert of {len(data)} records...")
                cursor.executemany(insert_sql, data)
                raw_conn.commit()
                
                self.logger.info(f"Ultra-fast export completed: {len(data)} records inserted successfully")
                
            finally:
                cursor.close()
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Ultra-fast export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to regular export method
            self.logger.info("Falling back to regular export method...")
            self._export_to_database(rec_df)

    def _export_to_database_super_fast(self, rec_df: pd.DataFrame):
        """Super-fast bulk export optimized for millions of records using BCP-style bulk operations."""
        try:
            self.logger.info(f"Super-fast export: {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            # For very large datasets, use optimized chunking with larger chunks
            super_chunk_size = 200000  # 200K records per chunk for super-fast operations
            successful_chunks = 0
            failed_chunks = 0
            total_chunks = (len(rec_df) + super_chunk_size - 1) // super_chunk_size
            
            # Get connection once and reuse
            conn = self.engine.connect()
            raw_conn = conn.connection.driver_connection
            cursor = raw_conn.cursor()
            
            try:
                # Enable all performance optimizations
                cursor.fast_executemany = True
                
                # Truncate table first
                cursor.execute("TRUNCATE TABLE NSR.user_rec_als")
                
                # Prepare bulk insert SQL
                insert_sql = """
                    INSERT INTO NSR.user_rec_als 
                    (customer_id, item_id, als_score_raw, als_score_mm, als_score_rank, source) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Process data in super-large chunks for maximum efficiency
                for i in range(0, len(rec_df), super_chunk_size):
                    chunk = rec_df.iloc[i:i + super_chunk_size]
                    chunk_num = i // super_chunk_size + 1
                    
                    for attempt in range(self.max_retries):
                        try:
                            # Convert to list of tuples for maximum speed
                            data = [tuple(row) for row in chunk[["customer_id", "item_id", "als_score_raw", "als_score_mm", "als_score_rank", "source"]].values]
                            
                            # Bulk insert with fast_executemany
                            cursor.executemany(insert_sql, data)
                            raw_conn.commit()
                            
                            self.logger.info(f"Super-fast chunk {chunk_num}/{total_chunks} exported successfully ({len(chunk)} rows)")
                            successful_chunks += 1
                            break
                            
                        except Exception as chunk_error:
                            if attempt < self.max_retries - 1:
                                self.logger.warning(f"Super-fast chunk {chunk_num} failed (attempt {attempt + 1}/{self.max_retries}): {str(chunk_error)}")
                                import time
                                time.sleep(1)  # Very short retry delay for super-fast operations
                            else:
                                self.logger.error(f"Super-fast chunk {chunk_num} failed after {self.max_retries} attempts")
                                failed_chunks += 1
                                raise chunk_error
                
                self.logger.info(f"Super-fast export completed: {successful_chunks} successful chunks, {failed_chunks} failed chunks")
                
            finally:
                # Clean up connections
                cursor.close()
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Super-fast export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to ultra-fast export method
            self.logger.info("Falling back to ultra-fast export method...")
            self._export_to_database_ultra_fast(rec_df)

    def _export_to_database_bulk_insert(self, rec_df: pd.DataFrame):
        """Ultimate performance: BULK INSERT using temporary CSV file for millions of records."""
        try:
            self.logger.info(f"BULK INSERT export: {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            import tempfile
            import os
            
            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
                temp_filename = temp_file.name
                
                # Write CSV data directly to file
                rec_df[["customer_id", "item_id", "als_score_raw", "als_score_mm", "als_score_rank", "source"]].to_csv(
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
                    cursor.execute("TRUNCATE TABLE NSR.user_rec_als")
                    
                    # Use BULK INSERT for maximum performance
                    bulk_insert_sql = f"""
                        BULK INSERT NSR.user_rec_als
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
                    
                    self.logger.info(f"BULK INSERT completed: {len(rec_df)} records inserted successfully")
                    
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
            self.logger.error(f"BULK INSERT export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to super-fast export method
            self.logger.info("Falling back to super-fast export method...")
            self._export_to_database_super_fast(rec_df)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Train ALS model and export recommendations")
    parser.add_argument("--tenant", required=True, help="Tenant identifier")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no actual execution)")
    
    args = parser.parse_args()
    
    try:
        if args.dry_run:
            logger.info(f"Dry run mode: Would train ALS model for tenant {args.tenant}")
            return
        
        # Create trainer and run
        trainer = ALSTrainer(args.tenant)
        trainer.train_and_export()
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()