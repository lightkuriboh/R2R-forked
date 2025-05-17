import os
import psycopg # Main Psycopg 3 import
from psycopg_pool import ConnectionPool # Psycopg 3 connection pooling
# from psycopg.rows import dict_row # Optional: If you want results as dicts directly from cursor
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)
if not logger.handlers: # Basic logging config if not set up by R2R
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Database Connection Parameters (from Environment Variables) ---
DB_HOST = os.getenv("TEPPIAI_DB_HOST")
DB_PORT = os.getenv("TEPPIAI_DB_PORT", "5432")
DB_NAME = os.getenv("TEPPIAI_DB_NAME")
DB_USER = os.getenv("TEPPIAI_DB_USER")
DB_PASSWORD_FILE = os.getenv("TEPPIAI_DB_PASSWORD_FILE")

_db_password: Optional[str] = None
if DB_PASSWORD_FILE:
    try:
        with open(DB_PASSWORD_FILE, 'r') as f:
            _db_password = f.read().strip()
    except IOError as e:
        logger.error(f"Could not read TeppiAI DB password from file {DB_PASSWORD_FILE}: {e}")
else:
    _db_password = os.getenv("TEPPIAI_DB_PASSWORD")

_connection_pool: Optional[ConnectionPool] = None

def initialize_teppiai_db_connection_pool():
    """
    Initializes the Psycopg 3 connection pool for the TeppiAI logging database.
    This should be called once when the R2R FastAPI application starts.
    """
    global _connection_pool
    if not all([DB_HOST, DB_NAME, DB_USER, _db_password]):
        logger.error(
            "TeppiAI logging DB connection parameters (TEPPIAI_DB_HOST, TEPPIAI_DB_NAME, "
            "TEPPIAI_DB_USER, and TEPPIAI_DB_PASSWORD_FILE or TEPPIAI_DB_PASSWORD) "
            "not fully configured. Chat logging to main TeppiAI DB will be disabled."
        )
        _connection_pool = None
        return

    conninfo = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={_db_password} connect_timeout=10"

    try:
        logger.info(f"Initializing TeppiAI logging DB connection pool for {DB_NAME} at {DB_HOST}:{DB_PORT}...")
        _connection_pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5, # Adjust as needed
            # open=False # Set to False if using with FastAPI lifespan to open it manually
        )
        # Test the pool by getting a connection
        with _connection_pool.connection() as conn:
            logger.info(f"Successfully established initial connection to TeppiAI logging DB: {conn.info.dbname} on {conn.info.host}")
        logger.info("TeppiAI logging DB connection pool initialized successfully.")

    except Exception as error:
        logger.error(f"Error while initializing TeppiAI logging DB connection pool: {error}")
        _connection_pool = None

def shutdown_teppiai_db_connection_pool():
    """
    Closes all connections in the TeppiAI logging DB connection pool.
    This should be called when the R2R application shuts down.
    """
    global _connection_pool
    if _connection_pool:
        try:
            logger.info("Closing TeppiAI logging DB connection pool...")
            _connection_pool.close() # For psycopg_pool.ConnectionPool (this is a blocking call)
            _connection_pool = None # Clear the global pool variable
            logger.info("TeppiAI logging DB connection pool closed successfully.")
        except Exception as e:
            logger.error(f"Error closing TeppiAI logging DB connection pool: {e}", exc_info=True)

def log_chat_interaction(
    session_id: str,
    customer_api_key: str,
    model_id_used: str,
    turn: int,
    role: str,
    content: str,
    retrieved_context: Optional[Dict[str, Any]] = None,
    feedback: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Logs a single chat interaction turn. Returns log_id or None.
    """
    if not _connection_pool:
        logger.warning("TeppiAI logging DB: Pool not available. Skipping chat log for session %s, turn %s.", session_id, turn)
        return None

    log_id = None
    sql = """
        INSERT INTO logs.chat_logs 
        (session_id, customer_api_key, model_id_used, turn, role, content, 
         retrieved_context, feedback, timestamp, metadata)
        VALUES (%(session_id)s, %(customer_api_key)s, %(model_id_used)s, %(turn)s, %(role)s, %(content)s, 
                %(retrieved_context)s, %(feedback)s, %(timestamp)s, %(metadata)s)
        RETURNING log_id;
    """
    params = {
        "session_id": session_id,
        "customer_api_key": customer_api_key,
        "model_id_used": model_id_used,
        "turn": turn,
        "role": role,
        "content": content,
        "retrieved_context": json.dumps(retrieved_context) if retrieved_context is not None else None,
        "feedback": feedback,
        "timestamp": datetime.now(timezone.utc),
        "metadata": json.dumps(metadata) if metadata is not None else None,
    }

    try:
        # `with` statement ensures connection is returned to pool
        with _connection_pool.connection() as conn:
            # `with` statement ensures cursor is closed
            with conn.cursor() as cur:
                cur.execute(sql, params)
                result = cur.fetchone()
                if result:
                    log_id = result[0]
                # conn.commit() is called automatically by 'with conn:' block if no exceptions
        
        if log_id:
            logger.info(f"TeppiAI logging DB: Logged chat interaction for session {session_id}, turn {turn}. Log ID: {log_id}")
        else:
            logger.warning(f"TeppiAI logging DB: Chat interaction logged for session {session_id}, turn {turn}, but no log_id returned.")
        return log_id
    except psycopg.Error as db_err: # Catch specific Psycopg errors
        logger.error(f"TeppiAI logging DB (Psycopg Error): Error logging chat interaction for session {session_id}, turn {turn}: {db_err}", exc_info=True)
        # No explicit rollback needed with `with conn:` if an exception occurs, it's handled.
        return None
    except Exception as e: # Catch other potential errors
        logger.error(f"TeppiAI logging DB (General Error): Error logging chat interaction for session {session_id}, turn {turn}: {e}", exc_info=True)
        return None

def update_chat_feedback(log_id: int, feedback_value: int) -> bool:
    """
    Updates feedback for a log entry. Returns True on success.
    """
    if not _connection_pool:
        logger.warning("TeppiAI logging DB: Pool not available. Skipping feedback update for log_id %s.", log_id)
        return False
    
    if feedback_value not in [-1, 0, 1]:
        logger.error(f"Invalid feedback value: {feedback_value} for log_id {log_id}. Must be -1, 0, or 1.")
        return False

    sql = """
        UPDATE logs.chat_logs 
        SET feedback = %(feedback_value)s 
        WHERE log_id = %(log_id)s;
    """
    params = {"feedback_value": feedback_value, "log_id": log_id}

    try:
        with _connection_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                updated_rows = cur.rowcount 
        
        if updated_rows > 0:
            logger.info(f"TeppiAI logging DB: Updated feedback for log_id {log_id} to {feedback_value}.")
            return True
        else:
            logger.warning(f"TeppiAI logging DB: No log entry found for log_id {log_id} or feedback value was unchanged.")
            return False
    except psycopg.Error as db_err:
        logger.error(f"TeppiAI logging DB (Psycopg Error): Error updating feedback for log_id {log_id}: {db_err}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"TeppiAI logging DB (General Error): Error updating feedback for log_id {log_id}: {e}", exc_info=True)
        return False

# --- Lifespan Integration for R2R's FastAPI app ---
# You'll need to integrate this into R2R's main FastAPI app startup/shutdown.
# Example (conceptual, to be adapted into R2R's actual main app file):
#
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# # Assuming this module is r2r_app.core.teppiai_logging_db
# from .teppiai_logging_db import initialize_teppiai_db_connection_pool, _connection_pool as logging_db_pool
#
# @asynccontextmanager
# async def r2r_app_lifespan(app: FastAPI):
#     print("R2R Main App: Lifespan startup...")
#     # R2R's own startup tasks (e.g., its own DB pools, loading configs)
#     # ...
#
#     # Initialize our TeppiAI logging DB pool
#     initialize_teppiai_db_connection_pool()
#     if logging_db_pool:
#         await logging_db_pool.open() # For psycopg_pool.ConnectionPool, you might need to open it if not done by default or if `open=False` in constructor
#                                      # Check psycopg_pool docs for best practice with FastAPI lifespan
#
#     print("R2R Main App: Startup complete.")
#     yield
#     print("R2R Main App: Lifespan shutdown...")
#     # R2R's own shutdown tasks
#     # ...
#
#     # Close our TeppiAI logging DB pool
#     if logging_db_pool:
#         await logging_db_pool.close() # Gracefully close all connections in the pool
#         print("TeppiAI logging DB connection pool closed.")
#     print("R2R Main App: Shutdown complete.")
#
# # For now, a simple attempt to initialize if not already done.
# if _connection_pool is None:
#     initialize_teppiai_db_connection_pool()
