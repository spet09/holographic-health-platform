"""
Database connection and management
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_models import Base, User, Analysis
from dotenv import load_dotenv
from datetime import datetime
import bcrypt
import json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./holographic_health.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database operations manager"""

    @staticmethod
    def save_user(db, user_data: dict):
        """Save user to database"""
        # Hash password
        password_hash = bcrypt.hashpw(
            user_data['password'].encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        user = User(
            username=user_data['username'],
            email=user_data['email'],
            phone=user_data.get('phone', ''),
            password_hash=password_hash
        )

        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate_user(db, username: str, password: str):
        """Authenticate user login"""
        user = db.query(User).filter(User.username == username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        return None

    @staticmethod
    def save_analysis(db, user_id: int, analysis_data: dict):
        """Save analysis results to database"""
        analysis = Analysis(
            user_id=user_id,
            results_data=json.dumps(analysis_data),
            life_force_percentage=analysis_data.get('life_force_percentage'),
            element_count=len(analysis_data.get('matches', [])),
            confidence_average=sum(m['confidence'] for m in analysis_data.get('matches', [])) / max(1,
                                                                                                    len(analysis_data.get(
                                                                                                        'matches', [])))
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis

    @staticmethod
    def get_user_analyses(db, user_id: int, limit: int = 50):
        """Get user's analysis history"""
        return db.query(Analysis).filter(
            Analysis.user_id == user_id
        ).order_by(Analysis.created_at.desc()).limit(limit).all()