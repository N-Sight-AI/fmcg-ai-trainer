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
import time
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
                SELECT
                    customer_id,
                    item_id,
                    weight,
                    sales_org_id
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
            if "sales_org_id" in df.columns:
                df["sales_org_id"] = df["sales_org_id"].astype(str)
            else:
                df["sales_org_id"] = "-1"
            
            df = df.dropna(subset=["customer_id", "item_id", "weight"]).astype({
                "customer_id": "int64",
                "item_id": "int64",
                "weight": "float32"
            })
            df["sales_org_id"] = df["sales_org_id"].fillna("-1").astype(str)
            
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
                SELECT customer_id,
                       item_id,
                       days_since_last,
                       med_cycle_days,
                       avg_cycle_days,
                       sales_org_id
                FROM NSR.customer_item_reorder_stats
                """,
                self.engine
            )
            df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
            df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
            df["days_since_last"] = pd.to_numeric(df["days_since_last"], errors="coerce")
            df["med_cycle_days"] = pd.to_numeric(df["med_cycle_days"], errors="coerce")
            df["avg_cycle_days"] = pd.to_numeric(df["avg_cycle_days"], errors="coerce")
            if "sales_org_id" in df.columns:
                df["sales_org_id"] = df["sales_org_id"].astype(str)
            else:
                df["sales_org_id"] = "-1"
            df = df.dropna(subset=["customer_id", "item_id"]).astype({"customer_id": "int64", "item_id": "int64"})
            df["sales_org_id"] = df["sales_org_id"].fillna("-1").astype(str)
            
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
            df = df.copy()
            df["customer_id"] = df["customer_id"].astype(str)
            
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
                self.logger.info(f"Preparing data for training (shape: {X.shape})...")
                item_user = X.T.tocsr().astype(np.float32)
                self.logger.info(f"Data prepared, creating ALS model (factors={self.factors}, reg={self.reg}, iters={self.iters})...")
                
                # Create and fit the model
                model = AlternatingLeastSquares(
                    factors=int(self.factors), 
                    regularization=float(self.reg), 
                    iterations=int(self.iters), 
                    use_gpu=bool(use_gpu)
                )
                self.logger.info("Starting model training...")
                self.logger.info(f"Data matrix shape: {item_user.shape}, nnz: {item_user.nnz}")
                self.logger.info(f"Matrix is {(item_user.shape[0], item_user.shape[1])} = (items, users)")
                self.logger.info("About to call model.fit() - this may take a while...")
                
                # Flush logs to ensure we see this before hang
                import sys
                if hasattr(sys, 'stdout'):
                    sys.stdout.flush()
                if hasattr(sys, 'stderr'):
                    sys.stderr.flush()
                
                try:
                    import time
                    start_time = time.time()
                    self.logger.info("Calling model.fit() now...")
                    
                    model.fit(item_user, show_progress=False)
                    
                    elapsed = time.time() - start_time
                    self.logger.info(f"model.fit() returned successfully after {elapsed:.2f} seconds")
                    self.logger.info("Model training completed successfully.")
                except Exception as fit_error:
                    self.logger.error(f"model.fit() raised exception: {type(fit_error).__name__}: {str(fit_error)}")
                    import traceback
                    self.logger.error(f"Fit traceback:\n{traceback.format_exc()}")
                    raise
                return model
            except Exception as e:
                self.logger.error(f"Failed to fit ALS model for tenant {self.tenant}: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                import traceback
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise
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
        df["sales_org_id"] = df["sales_org_id"].astype(str)
        df["als_score_mm"] = 0.0
        group_cols = ["customer_id", "sales_org_id"]
        
        # Handle backfill recommendations first
        backfill_mask = df["source"] == "backfill"
        df.loc[backfill_mask, "als_score_mm"] = self.backfill_floor
        
        # Handle ALS recommendations
        als_mask = df["source"] == "als"
        if not als_mask.any():
            return df
        
        # Calculate group sizes for each customer/sales_org pair
        counts_df = (
            df.loc[als_mask, group_cols]
            .assign(_count=1)
            .groupby(group_cols)["_count"]
            .sum()
            .rename("customer_item_count")
            .reset_index()
        )
        df = df.merge(counts_df, on=group_cols, how="left")
        df["customer_item_count"] = df["customer_item_count"].fillna(0)
        
        # Handle customers with few items (rank-based normalization)
        few_items_mask = als_mask & (df["customer_item_count"] < min_items)
        denom_few = np.maximum(df.loc[few_items_mask, "customer_item_count"] - 1, 1)
        if few_items_mask.any():
            df.loc[few_items_mask, "als_score_mm"] = (
                1.0 - (df.loc[few_items_mask, "als_score_rank"] - 1) / denom_few
            )
        
        # Handle customers with sufficient items (robust min-max normalization)
        sufficient_items_mask = als_mask & ~few_items_mask
        if sufficient_items_mask.any():
            quantile_df = (
                df.loc[sufficient_items_mask]
                .groupby(group_cols)["als_score_raw"]
                .quantile([lo / 100, hi / 100])
                .unstack()
                .rename(columns={lo / 100: "lo_val", hi / 100: "hi_val"})
            )
            quantile_df["span"] = quantile_df["hi_val"] - quantile_df["lo_val"]
            
            df = df.merge(quantile_df, on=group_cols, how="left")
            
            span_too_small_mask = sufficient_items_mask & (df["span"] < min_span)
            if span_too_small_mask.any():
                denom_small = np.maximum(df.loc[span_too_small_mask, "customer_item_count"] - 1, 1)
                df.loc[span_too_small_mask, "als_score_mm"] = (
                    1.0 - (df.loc[span_too_small_mask, "als_score_rank"] - 1) / denom_small
                )
            
            minmax_mask = sufficient_items_mask & (df["span"] >= min_span)
            if minmax_mask.any():
                minmax_scores = np.clip(
                    (df.loc[minmax_mask, "als_score_raw"] - df.loc[minmax_mask, "lo_val"])
                    / df.loc[minmax_mask, "span"],
                    0,
                    1,
                )
                denom_minmax = np.maximum(df.loc[minmax_mask, "customer_item_count"] - 1, 1)
                rank_scores = 1.0 - (df.loc[minmax_mask, "als_score_rank"] - 1) / denom_minmax
                df.loc[minmax_mask, "als_score_mm"] = (1 - rank_blend) * minmax_scores + rank_blend * rank_scores
            
            df = df.drop(columns=["lo_val", "hi_val", "span"], errors="ignore")
        
        df = df.drop(columns=["customer_item_count"], errors="ignore")
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
            interactions_df["sales_org_id"] = interactions_df["sales_org_id"].fillna("-1").astype(str)
            interactions_df["customer_sales_org_key"] = (
                interactions_df["customer_id"].astype(str) + "|" + interactions_df["sales_org_id"]
            )
            
            self.logger.info("Step 2: Loading policy data...")
            policy_df = self.load_policy()
            
            self.logger.info("Step 3: Loading reorder stats...")
            reorder_stats_df = self.load_reorder_stats()
            
            self.logger.info("Step 4: Loading inventory data...")
            inventory_df = self.load_inventory()
            
            # Build interaction matrix
            self.logger.info("Step 5: Building interaction matrix...")
            matrix_input_df = interactions_df[["customer_sales_org_key", "item_id", "weight"]].rename(
                columns={"customer_sales_org_key": "customer_id"}
            )
            X, u_uniques, i_uniques = self.build_matrix(matrix_input_df)
            
            # Fit ALS model
            self.logger.info("Step 6: Training ALS model...")
            model = self.fit_als_autoorient(X)
            
            # Generate recommendations
            self.logger.info("Step 7: Generating recommendations...")
            
            # Get user factors and normalize
            # Note: Since we passed X.T to model.fit():
            # - model.user_factors corresponds to rows of X.T (which are items)
            # - model.item_factors corresponds to columns of X.T (which are users)
            # So we need to swap them!
            user_factors = self.l2_normalize(model.item_factors)  # Users are in item_factors
            item_factors = self.l2_normalize(model.user_factors)   # Items are in user_factors
            
            # Compute recommendations
            # user_factors shape: (n_users, factors)
            # item_factors shape: (n_items, factors)
            # rec_scores shape: (n_users, n_items)
            rec_scores = user_factors @ item_factors.T
            
            # Verify dimensions match
            self.logger.info(f"rec_scores shape: {rec_scores.shape}, expected: ({len(u_uniques)}, {len(i_uniques)})")
            assert rec_scores.shape[0] == len(u_uniques), f"User dimension mismatch: {rec_scores.shape[0]} != {len(u_uniques)}"
            assert rec_scores.shape[1] == len(i_uniques), f"Item dimension mismatch: {rec_scores.shape[1]} != {len(i_uniques)}"
            
            # Create recommendations dataframe using vectorized operations
            self.logger.info(f"Creating recommendations for {len(u_uniques)} users with top-{self.topk} items each...")
            
            # Vectorized approach: get top-k indices for all users at once
            # Ensure topk doesn't exceed the number of available items
            actual_topk = min(self.topk, rec_scores.shape[1])
            if actual_topk < self.topk:
                self.logger.warning(f"Reducing topk from {self.topk} to {actual_topk} due to insufficient items for tenant {self.tenant}")
            
            top_k_indices = np.argpartition(-rec_scores, actual_topk-1, axis=1)[:, :actual_topk]
            
            # Sort only the top-k items for each user by their scores
            top_k_scores = np.take_along_axis(rec_scores, top_k_indices, axis=1)
            sorted_inds = np.argsort(-top_k_scores, axis=1)
            top_k_indices = np.take_along_axis(top_k_indices, sorted_inds, axis=1)
            top_k_scores = np.take_along_axis(top_k_scores, sorted_inds, axis=1)
            
            # Create arrays for all recommendations at once
            num_recs = len(u_uniques) * actual_topk
            customer_keys = np.repeat(np.asarray(u_uniques, dtype=str), actual_topk)
            partitioned_keys = np.char.partition(customer_keys, "|")
            customer_ids = partitioned_keys[:, 0].astype(np.int64)
            sales_org_ids = np.where(partitioned_keys[:, 1] == "", "-1", partitioned_keys[:, 2])
            item_ids = i_uniques[top_k_indices.flatten()]
            scores = top_k_scores.flatten()
            ranks = np.tile(np.arange(1, actual_topk + 1), len(u_uniques))
            
            # Create DataFrame directly from arrays
            rec_df = pd.DataFrame({
                "customer_id": customer_ids,
                "sales_org_id": sales_org_ids,
                "item_id": item_ids,
                "als_score_raw": scores.astype(float),
                "als_score_rank": ranks,
                "source": "als"
            })
            rec_df["sales_org_id"] = rec_df["sales_org_id"].astype(str)
            
            self.logger.info(f"Generated {len(rec_df)} ALS recommendations")
            
            # Add backfill recommendations for customers with few ALS recommendations
            self.logger.info("Step 8: Adding backfill recommendations...")
            customer_counts = rec_df.groupby(["customer_id", "sales_org_id"]).size()
            customers_needing_backfill = customer_counts[customer_counts < 10]
            
            if not customers_needing_backfill.empty:
                self.logger.info(f"Adding backfill recommendations for {len(customers_needing_backfill)} customer/sales_org pairs")
                
                # Get items that are due for reorder
                due_items = reorder_stats_df[reorder_stats_df.apply(self.is_due, axis=1)].copy()
                due_items["sales_org_id"] = due_items["sales_org_id"].astype(str)
                
                # Vectorized backfill approach
                backfill_recs = []
                for (customer_id, sales_org_id), _ in customers_needing_backfill.items():
                    customer_due_items = due_items[
                        (due_items["customer_id"] == customer_id) &
                        (due_items["sales_org_id"] == sales_org_id)
                    ]
                    if len(customer_due_items) > 0:
                        top_items = customer_due_items.head(10)
                        backfill_recs.append(pd.DataFrame({
                            "customer_id": customer_id,
                            "sales_org_id": sales_org_id,
                            "item_id": top_items["item_id"].values,
                            "als_score_raw": 0.0,
                            "als_score_rank": 999,
                            "source": "backfill"
                        }))
                
                if backfill_recs:
                    backfill_df = pd.concat(backfill_recs, ignore_index=True)
                    rec_df = pd.concat([rec_df, backfill_df], ignore_index=True)
                    self.logger.info(f"Added {len(backfill_df)} backfill recommendations")
            
            # Attach sales organization information
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
            initial_chunk = self.chunk_size if self.chunk_size > 0 else 10000
            initial_chunk = max(2000, min(initial_chunk, len(rec_df), 20000))
            self._bulk_insert_with_strategy(rec_df, initial_chunk, min_chunk_size=1000, strategy="bulk")
        except Exception as e:
            self.logger.error(f"Failed to export recommendations for tenant {self.tenant}: {str(e)}")
            raise

    def _export_to_database_ultra_fast(self, rec_df: pd.DataFrame):
        """Ultra-fast bulk export using SQLAlchemy bulk operations for very large datasets."""
        try:
            self.logger.info(f"Ultra-fast export: {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            initial_chunk = self.chunk_size * 2 if self.chunk_size > 0 else 40000
            initial_chunk = max(2000, min(initial_chunk, len(rec_df), 50000))
            self._bulk_insert_with_strategy(rec_df, initial_chunk, min_chunk_size=1500, strategy="ultra-fast")
        except Exception as e:
            self.logger.error(f"Ultra-fast export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to regular export method
            self.logger.info("Falling back to regular export method...")
            self._export_to_database(rec_df)

    def _export_to_database_super_fast(self, rec_df: pd.DataFrame):
        """Super-fast bulk export optimized for millions of records using BCP-style bulk operations."""
        try:
            self.logger.info(f"Super-fast export: {len(rec_df)} recommendations to database for tenant {self.tenant}")
            
            initial_chunk = self.chunk_size * 3 if self.chunk_size > 0 else 75000
            initial_chunk = max(5000, min(initial_chunk, len(rec_df), 100000))
            self._bulk_insert_with_strategy(rec_df, initial_chunk, min_chunk_size=2500, strategy="super-fast")
        except Exception as e:
            self.logger.error(f"Super-fast export failed for tenant {self.tenant}: {str(e)}")
            # Fallback to ultra-fast export method
            self.logger.info("Falling back to ultra-fast export method...")
            self._export_to_database_ultra_fast(rec_df)

    def _bulk_insert_with_strategy(
        self,
        rec_df: pd.DataFrame,
        initial_chunk_size: int,
        min_chunk_size: int,
        strategy: str,
    ) -> None:
        """Common bulk insert implementation with adaptive chunk sizing and retry support."""
        # Ensure sales_org_id column exists before processing
        if "sales_org_id" not in rec_df.columns:
            rec_df = rec_df.copy()
            rec_df["sales_org_id"] = "-1"
        else:
            rec_df = rec_df.copy()
            rec_df["sales_org_id"] = rec_df["sales_org_id"].fillna("-1").astype(str)

        columns = [
            "customer_id",
            "item_id",
            "als_score_raw",
            "als_score_mm",
            "als_score_rank",
            "source",
            "sales_org_id",
        ]
        dtype_map = {
            "customer_id": "int64",
            "item_id": "int64",
            "als_score_raw": "float64",
            "als_score_mm": "float64",
            "als_score_rank": "float64",
        }

        min_chunk_size = max(500, min_chunk_size)
        total_rows = len(rec_df)
        chunk_size = max(min_chunk_size, min(initial_chunk_size, total_rows)) if total_rows else min_chunk_size

        insert_sql = """
            INSERT INTO NSR.user_rec_als
            (customer_id, item_id, als_score_raw, als_score_mm, als_score_rank, source, sales_org_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        conn = None
        cursor = None
        inserted_rows = 0
        successful_chunks = 0
        failed_chunks = 0
        start_time = time.time()

        try:
            conn = self.engine.raw_connection()
            raw_conn = conn
            raw_conn.autocommit = False
            cursor = raw_conn.cursor()

            # Enable driver optimisations when available
            if hasattr(cursor, "fast_executemany"):
                cursor.fast_executemany = True
            if hasattr(cursor, "timeout") and self.command_timeout:
                try:
                    cursor.timeout = self.command_timeout
                except Exception:
                    pass
            if hasattr(cursor, "arraysize"):
                cursor.arraysize = chunk_size

            # Always truncate before inserting new recommendations
            cursor.execute("TRUNCATE TABLE NSR.user_rec_als")
            raw_conn.commit()

            if total_rows == 0:
                self.logger.info(f"{strategy.capitalize()} export: no rows to insert after truncate.")
                return

            position = 0
            chunk_index = 0

            while position < total_rows:
                remaining = total_rows - position
                current_chunk_size = min(chunk_size, remaining)
                chunk = rec_df.iloc[position: position + current_chunk_size].copy()

                if "sales_org_id" in chunk.columns:
                    chunk["sales_org_id"] = chunk["sales_org_id"].fillna("-1").astype(str)
                else:
                    chunk["sales_org_id"] = "-1"

                for col, dtype in dtype_map.items():
                    chunk[col] = chunk[col].astype(dtype, copy=False)
                chunk["source"] = chunk["source"].astype(str)

                attempts = 0
                backoff_seconds = 1.0

                while attempts < self.max_retries:
                    rows = [tuple(row) for row in chunk[columns].itertuples(index=False, name=None)]
                    chunk_label = chunk_index + 1

                    try:
                        begin = time.time()
                        self.logger.debug(
                            "%s chunk %s: inserting %s rows (position %s)",
                            strategy,
                            chunk_label,
                            len(rows),
                            position,
                        )

                        cursor.executemany(insert_sql, rows)
                        raw_conn.commit()

                        duration = time.time() - begin
                        inserted_rows += len(rows)
                        chunk_index += 1
                        successful_chunks += 1

                        self.logger.info(
                            "%s chunk %s committed (%s rows in %.2fs, cumulative %s/%s)",
                            strategy.capitalize(),
                            chunk_label,
                            len(rows),
                            duration,
                            inserted_rows,
                            total_rows,
                        )

                        position += len(rows)
                        break

                    except Exception as chunk_error:
                        raw_conn.rollback()
                        attempts += 1

                        self.logger.warning(
                            "%s chunk %s failed on attempt %s/%s with chunk_size=%s: %s",
                            strategy.capitalize(),
                            chunk_label,
                            attempts,
                            self.max_retries,
                            len(rows),
                            chunk_error,
                        )

                        # Reduce chunk size to alleviate pressure if possible
                        if chunk_size > min_chunk_size and len(rows) > min_chunk_size:
                            old_chunk_size = chunk_size
                            chunk_size = max(min_chunk_size, chunk_size // 2)
                            self.logger.info(
                                "%s chunk size reduced from %s to %s rows due to failure; retrying same position %s",
                                strategy.capitalize(),
                                old_chunk_size,
                                chunk_size,
                                position,
                            )
                            current_chunk_size = min(chunk_size, remaining)
                            chunk = rec_df.iloc[position: position + current_chunk_size].copy()
                            if "sales_org_id" in chunk.columns:
                                chunk["sales_org_id"] = chunk["sales_org_id"].fillna("-1").astype(str)
                            else:
                                chunk["sales_org_id"] = "-1"
                            for col, dtype in dtype_map.items():
                                chunk[col] = chunk[col].astype(dtype, copy=False)
                            chunk["source"] = chunk["source"].astype(str)
                            backoff_seconds = 1.0
                            # continue with smaller chunk without increasing attempts counter
                            continue

                        if attempts < self.max_retries:
                            sleep_time = min(5.0, backoff_seconds)
                            self.logger.info(
                                "Retrying %s chunk %s in %.1fs...",
                                strategy,
                                chunk_label,
                                sleep_time,
                            )
                            time.sleep(sleep_time)
                            backoff_seconds = min(5.0, backoff_seconds * 2)
                        else:
                            failed_chunks += 1
                            raise

            raw_conn.commit()

            elapsed = time.time() - start_time
            self.logger.info(
                "%s export completed: %s rows inserted in %.2fs (successful chunks=%s, failed chunks=%s, final chunk size=%s).",
                strategy.capitalize(),
                inserted_rows,
                elapsed,
                successful_chunks,
                failed_chunks,
                chunk_size,
            )

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

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
                rec_df[["customer_id", "item_id", "als_score_raw", "als_score_mm", "als_score_rank", "source", "sales_org_id"]].to_csv(
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