"""
Database Manager for Smart Parking System.
Uses SQLite for persistent storage of vehicle check-in / check-out sessions,
license plates, and driver face embeddings.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np


class ParkingDatabase:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Create necessary tables with indexing for high performance."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parking_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    embedding BLOB,
                    time_in TEXT NOT NULL,
                    time_out TEXT,
                    duration_minutes REAL,
                    match_score REAL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_plate_active ON parking_sessions (plate, active)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_active ON parking_sessions (active)"
            )

    def register(self, person_name: str, plate: str, embedding: Optional[np.ndarray] = None) -> int:
        """
        Register a new vehicle entry (Check-in).
        """
        emb_bytes = None
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32)
            emb_bytes = emb.tobytes()

        time_in_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO parking_sessions(person_name, plate, embedding, time_in, active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (person_name, plate.strip().upper(), emb_bytes, time_in_str),
            )
            session_id = int(cursor.lastrowid)

        # If name was empty, update to Session ID
        if not person_name or person_name.strip() == "":
            self.update_person_name(session_id, f"Guest #{session_id}")

        return session_id

    def update_person_name(self, session_id: int, person_name: str) -> None:
        """Update driver name for a session."""
        with self.conn:
            self.conn.execute(
                "UPDATE parking_sessions SET person_name = ? WHERE id = ?",
                (person_name, session_id),
            )

    def find_active_by_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        """
        Find an active parking session by license plate string.
        """
        clean_plate = plate.strip().upper().replace(" ", "").replace("-", "").replace(".", "")
        rows = self.conn.execute(
            """
            SELECT id, person_name, plate, embedding, time_in
            FROM parking_sessions
            WHERE active = 1
            """
        ).fetchall()

        for row in rows:
            db_plate = row["plate"].strip().upper().replace(" ", "").replace("-", "").replace(".", "")
            if db_plate == clean_plate:
                emb = None
                if row["embedding"] is not None:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                return {
                    "id": row["id"],
                    "person_name": row["person_name"],
                    "plate": row["plate"],
                    "embedding": emb,
                    "time_in": row["time_in"],
                }
        return None

    def find_best_face_active(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Search all active sessions to find the closest matching driver face.
        Calculates Cosine Similarity against stored embeddings.
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None
        query = query / query_norm

        rows = self.conn.execute(
            """
            SELECT id, person_name, plate, embedding, time_in
            FROM parking_sessions
            WHERE active = 1 AND embedding IS NOT NULL
            """
        ).fetchall()

        best_session = None
        best_score = -1.0

        for row in rows:
            blob = row["embedding"]
            if blob is None:
                continue
            saved = np.frombuffer(blob, dtype=np.float32)
            saved_norm = np.linalg.norm(saved)
            if saved_norm == 0 or saved.shape != query.shape:
                continue

            saved = saved / saved_norm
            score = float(np.dot(query, saved))

            if score > best_score:
                best_score = score
                best_session = {
                    "id": row["id"],
                    "person_name": row["person_name"],
                    "plate": row["plate"],
                    "time_in": row["time_in"],
                    "score": score,
                }

        return best_session

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two face embeddings."""
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0 or v1.shape != v2.shape:
            return 0.0
        return float(np.dot(v1 / n1, v2 / n2))

    def close_session(self, session_id: int, match_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Close an active parking session (Check-out).
        Calculates total parked duration in minutes.
        """
        row = self.conn.execute(
            "SELECT time_in FROM parking_sessions WHERE id = ?", (session_id,)
        ).fetchone()

        if not row:
            raise ValueError(f"Session #{session_id} not found.")

        time_in_dt = datetime.strptime(row["time_in"], "%Y-%m-%d %H:%M:%S")
        now_dt = datetime.now()
        duration_minutes = round((now_dt - time_in_dt).total_seconds() / 60.0, 2)
        time_out_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        with self.conn:
            self.conn.execute(
                """
                UPDATE parking_sessions
                SET active = 0, time_out = ?, duration_minutes = ?, match_score = ?
                WHERE id = ?
                """,
                (time_out_str, duration_minutes, match_score, session_id),
            )

        return {
            "session_id": session_id,
            "time_in": row["time_in"],
            "time_out": time_out_str,
            "duration_minutes": duration_minutes,
            "match_score": match_score,
        }

    def active_count(self) -> int:
        """Get current number of parked vehicles."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM parking_sessions WHERE active = 1"
        ).fetchone()
        return int(row[0]) if row else 0

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recent sessions."""
        rows = self.conn.execute(
            """
            SELECT id, person_name, plate, time_in, time_out, duration_minutes, match_score, active
            FROM parking_sessions
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close SQLite connection."""
        self.conn.close()
