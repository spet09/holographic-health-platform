"""
Database models and connection management
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    phone = Column(String(20), nullable=True)
    password_hash = Column(String(255), nullable=False)
    subscription_type = Column(String(20), default="free")
    subscription_status = Column(String(20), default="active")
    stripe_customer_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    analyses_count = Column(Integer, default=0)
    monthly_analyses_reset = Column(DateTime, default=datetime.utcnow)

    # Relationships
    health_profiles = relationship("HealthProfile", back_populates="user")
    analyses = relationship("Analysis", back_populates="user")
    payments = relationship("Payment", back_populates="user")


class HealthProfile(Base):
    __tablename__ = "health_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    height = Column(Float, nullable=True)  # cm
    weight = Column(Float, nullable=True)  # kg
    medical_conditions = Column(Text, nullable=True)  # JSON string
    medications = Column(Text, nullable=True)
    allergies = Column(Text, nullable=True)
    lifestyle_data = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="health_profiles")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_type = Column(String(50), default="spectral")
    image_filename = Column(String(255), nullable=True)
    image_data = Column(Text, nullable=True)  # Base64 encoded
    results_data = Column(Text, nullable=False)  # JSON string
    life_force_percentage = Column(Float, nullable=True)
    element_count = Column(Integer, nullable=True)
    confidence_average = Column(Float, nullable=True)
    analysis_duration = Column(Float, nullable=True)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="analyses")


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stripe_payment_intent_id = Column(String(100), unique=True, nullable=False)
    amount = Column(Integer, nullable=False)  # cents
    currency = Column(String(3), default="usd")
    subscription_plan = Column(String(20), nullable=False)
    payment_status = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="payments")


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)