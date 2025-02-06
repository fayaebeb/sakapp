import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import jwt
from jose.exceptions import JWTError
import requests
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

# =============================================================================
# Environment Variables and Configuration
# =============================================================================
# (Set these in your Vercel project settings)
SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")  # For production, use your Supabase/Neon URL

# Auth0 configuration (set these via Vercel’s environment variables)
AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "your-auth0-domain.auth0.com")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "your-api-audience")
ALGORITHMS = ["RS256"]

# Langflow API configuration (set these via environment variables)
BASE_API_URL = os.environ.get("BASE_API_URL", "https://api.langflow.astra.datastax.com")
LANGFLOW_ID = os.environ.get("LANGFLOW_ID", "2e964804-1fee-4340-bb22-099f1e785ec1")
FLOW_ID = os.environ.get("FLOW_ID", "edf93eff-1384-4865-bc9e-b7bbcbd9ed1a")
APPLICATION_TOKEN = os.environ.get("APPLICATION_TOKEN", "your_application_token")
ENDPOINT = os.environ.get("ENDPOINT", "")

# =============================================================================
# Database Setup (SQLAlchemy)
# =============================================================================
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Model for storing chat history
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # This will store the Auth0 user ID (the "sub" claim)
    message = Column(Text)
    response = Column(Text)

# Create tables (for local testing; in production use migration tools)
Base.metadata.create_all(bind=engine)

# Dependency: Database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# Auth0 JWT Verification (for protecting endpoints)
# =============================================================================
def get_auth0_public_key():
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    jwks = requests.get(jwks_url).json()
    return jwks

def verify_token(token: str):
    jwks = get_auth0_public_key()
    unverified_header = jwt.get_unverified_header(token)
    rsa_key = {}
    for key in jwks["keys"]:
        if key["kid"] == unverified_header.get("kid"):
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"]
            }
            break
    if rsa_key:
        try:
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=ALGORITHMS,
                audience=AUTH0_AUDIENCE,
                issuer=f"https://{AUTH0_DOMAIN}/"
            )
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    raise HTTPException(status_code=401, detail="Unable to find appropriate key")

# Dependency: Extract and verify the JWT from the Authorization header
auth_scheme = HTTPBearer()

def get_current_user(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    payload = verify_token(token.credentials)
    return payload  # This payload contains Auth0 claims (e.g. "sub")

# =============================================================================
# Chat API: Call Langflow and Store Chat History
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    tweaks: Optional[dict] = None
    endpoint: Optional[str] = None
    output_type: str = "chat"
    input_type: str = "chat"

def run_flow(
    message: str,
    endpoint: str,
    output_type: str = "chat",
    input_type: str = "chat",
    tweaks: Optional[dict] = None,
) -> dict:
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    if tweaks:
        payload["tweaks"] = tweaks
    headers = {
        "Authorization": f"Bearer {APPLICATION_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()

# =============================================================================
# FastAPI Application Initialization
# =============================================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Example protected homepage (for testing)
@app.get("/", response_class=HTMLResponse)
def homepage(current_user: dict = Depends(get_current_user)):
    # Display a simple welcome message using the Auth0 "sub" claim as the user ID
    return f"<h1>Welcome, {current_user.get('sub')}</h1><p>This is your chatbot application.</p>"

# Chat endpoint (protected by Auth0)
@app.post("/chat")
def chat_endpoint(chat_request: ChatRequest, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    # Call Langflow API
    api_response = run_flow(
        message=chat_request.message,
        endpoint=chat_request.endpoint or ENDPOINT,
        output_type=chat_request.output_type,
        input_type=chat_request.input_type,
        tweaks=chat_request.tweaks,
    )
    # Extract the AI response (adjust the extraction based on your Langflow response structure)
    ai_message = (
        api_response.get("outputs", [{}])[0]
        .get("outputs", [{}])[0]
        .get("results", {})
        .get("message", {})
        .get("data", {})
        .get("text", "")
    )
    # Store the chat history (using the Auth0 user id from the token payload, typically in the "sub" claim)
    user_id = current_user.get("sub")
    chat_history = ChatHistory(user_id=user_id, message=chat_request.message, response=ai_message)
    db.add(chat_history)
    db.commit()
    return api_response
