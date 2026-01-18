#!/usr/bin/env python3
"""API-based quiz runner that evaluates cloud LLM APIs (OpenAI, Google Gemini, Anthropic Claude, x.ai Grok, Together AI)."""

from __future__ import annotations
import csv
import logging
import sys
import re
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import suppress
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pymongo import MongoClient

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API client imports
try:
    import openai
except ImportError:
    openai = None
    print("Warning: openai not installed. Install with: pip install openai")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic not installed. Install with: pip install anthropic")

# json is used for API responses
import json

# ────────── config ──────────
QUESTIONS_PER_FIELD = 1_000
MAX_REPROMPTS = 2
TIMEOUT_SEC = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0
# Increased retries for rate limit errors specifically
MAX_RETRIES_RATE_LIMIT = 7

# Same validation constants as quiz.py
VALID_ANSWERS = {"true", "false", "possibly true", "possibly false", "unknown"}
TRUE_SET, FALSE_SET = {"true", "possibly true"}, {"false", "possibly false"}
ALIAS = {"possbilytrue": "possibly true", "possiblyfalse": "possibly false"}
NUMERIC_CHOICES = {
    "1": "true",
    "2": "false",
    "3": "possibly true",
    "4": "possibly false",
    "5": "unknown",
}
FILLER_PHRASES = {
    "possible truth": "possibly true",
    "possibly truth": "possibly true",
    "possible true": "possibly true",
    "it is false": "false",
    "the answer is no": "false",
    "answer is no": "false",
    "answer is false": "false",
    "possible false": "possibly false",
    "possibly false": "possibly false",
    "not sure": "unknown",
    "can't tell": "unknown",
    "cannot tell": "unknown",
    "no idea": "unknown",
}
FILLER_SYNONYMS = {
    "true": {"yes", "yeah", "yep"},
    "false": {"no", "nope", "nah", "negative", "incorrect"},
    "possibly true": {"maybe", "probably", "likely", "perhaps"},
    "possibly false": {"unlikely"},
    "unknown": {"unknown", "unsure", "uncertain", "indeterminate"},
}
FILLER_HINTS = (
    "the paper",
    "here is",
    "here's",
    "the question is",
    "to answer this question",
    "what a fascinating question",
    "the answer to this question",
    "the answer to the above question",
    "this paper",
    "the abstract",
)

REPROMPT_TEMPLATE = (
    "\n\nYour previous answer \"{answer}\" was invalid because {reason}. "
    "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown. "
    "Do not start with filler phrases such as \"here is\" or \"the answer is\", and do not write any sentences. "
    "Copy the word exactly as spelled above with no punctuation or extras. "
    "If you are unsure, reply with unknown. Never leave the reply blank."
)

FIELD_MAP = {
    "field_11": "agricultural-and-biological-sciences",
    "field_12": "arts-and-humanities",
    "field_13": "biochemistry-genetics-and-molecular-biology",
    "field_14": "business-management-and-accounting",
    "field_15": "chemical-engineering",
    "field_16": "chemistry",
    "field_17": "computer-science",
    "field_18": "decision-sciences",
    "field_19": "earth-and-planetary-sciences",
    "field_20": "economics-econometrics-and-finance",
    "field_21": "energy",
    "field_22": "engineering",
    "field_23": "environmental-science",
    "field_24": "immunology-and-microbiology",
    "field_25": "materials-science",
    "field_26": "mathematics",
    "field_27": "medicine",
    "field_28": "neuroscience",
    "field_29": "nursing",
    "field_30": "pharmacology-toxicology-and-pharmaceutics",
    "field_31": "physics-and-astronomy",
    "field_32": "psychology",
    "field_33": "social-sciences",
    "field_34": "veterinary",
    "field_35": "dentistry",
    "field_36": "health-professions",
}
DATABASES = list(FIELD_MAP.keys())

MONGO_URI = "mongodb://localhost:27017/"
LOG_PATH = Path("quiz_apis.log")
CSV_PATH = Path("results-apis.csv")

# ────────── scoring constants ──────────
CORRECT_SCORE = 0.1   # 1000 correct answers → 100 points
INCORRECT_SCORE = -0.1

# Global tracking
GLOBAL_SALVAGE_TOTAL: int = 0
GLOBAL_SALVAGE_BY_MODEL: Dict[str, int] = {}
GLOBAL_SALVAGE_BY_FIELD: Dict[str, int] = defaultdict(int)

# ────────── logging ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
log, log_warn, log_err = logger.info, logger.warning, logger.error

# ────────── API Client Initialization ──────────
OPENAI_CLIENT = None
GEMINI_CLIENT = None
ANTHROPIC_CLIENT = None
XAI_API_KEY = None
TOGETHER_API_KEY = None
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

def initialize_api_clients():
    """Initialize API clients with keys from environment."""
    global OPENAI_CLIENT, GEMINI_CLIENT, ANTHROPIC_CLIENT, XAI_API_KEY, TOGETHER_API_KEY
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai and openai_key:
        try:
            OPENAI_CLIENT = openai.OpenAI(api_key=openai_key, timeout=TIMEOUT_SEC)
            log("✅ OpenAI client initialized")
        except Exception as e:
            log_err(f"Failed to initialize OpenAI client: {e}")
    else:
        log_warn("OpenAI API key not found or openai not installed")
    
    # Google Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if genai and google_key:
        try:
            genai.configure(api_key=google_key)
            GEMINI_CLIENT = genai
            log("✅ Google Gemini client initialized")
        except Exception as e:
            log_err(f"Failed to initialize Gemini client: {e}")
    else:
        log_warn("Google API key not found or google-generativeai not installed")
    
    # Anthropic Claude
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic and anthropic_key:
        try:
            ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=anthropic_key, timeout=TIMEOUT_SEC * 1000)  # milliseconds
            log("✅ Anthropic Claude client initialized")
        except Exception as e:
            log_err(f"Failed to initialize Anthropic client: {e}")
    else:
        log_warn("Anthropic API key not found or anthropic not installed")
    
    # x.ai Grok
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    if XAI_API_KEY:
        log("✅ x.ai API key loaded")
    else:
        log_warn("x.ai API key not found")
    
    # Together AI (for Llama 4 Maverick)
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        TOGETHER_API_KEY = together_key
        log("✅ Together AI API key loaded")
    else:
        log_warn("Together AI API key not found. Get one at: https://www.together.ai/models/llama-4-maverick")

# ────────── Model Discovery ──────────
def discover_openai_models() -> List[str]:
    """Return GPT-5.1 with reasoning_effort=none for non-reasoning tasks."""
    # GPT-5.1 with reasoning_effort=none - most powerful non-reasoning model
    # Order of preference: GPT-5.1, then fallbacks
    preferred_models = [
        "gpt-5.1",               # GPT-5.1 with reasoning_effort=none (most powerful)
        "gpt-5.1-2025-04-14",    # Specific GPT-5.1 version
        "gpt-4.1",                # Fallback to GPT-4.1
        "gpt-4.1-2025-04-14",    # Specific GPT-4.1 version
        "gpt-4o",                 # Fallback to GPT-4o
        "gpt-4-turbo",            # Fallback to GPT-4-turbo
    ]
    
    # Exclude reasoning models and other unwanted models
    reasoning_indicators = ["o1", "o3", "o4", "reasoning", "think"]
    exclude_indicators = ["embed", "code", "audio", "realtime", "search", "transcribe"] + reasoning_indicators
    
    if not OPENAI_CLIENT:
        # Return GPT-5.1 as fallback
        model = preferred_models[0]  # GPT-5.1
        log(f"OpenAI client not initialized, using GPT-5.1 with reasoning_effort=none: {model}")
        return [model]
    
    # Try to verify which models are available
    available_models = []
    try:
        model_list = OPENAI_CLIENT.models.list()
        available_model_ids = {model.id for model in model_list.data}
        
        # Return the first preferred model that's available
        for model in preferred_models:
            if model in available_model_ids:
                available_models = [model]
                log(f"Using GPT-5.1 with reasoning_effort=none: {model}")
                break
        
        # If none of the preferred models are available, search for GPT-5.1, GPT-4.1, GPT-4o, or GPT-4
        if not available_models:
            for model in model_list.data:
                model_id = model.id.lower()
                # Exclude reasoning models and other unwanted types
                if any(exclude in model_id for exclude in exclude_indicators):
                    continue
                # Prioritize GPT-5.1, then GPT-4.1, GPT-4o, GPT-4-turbo, GPT-4 (but NOT o1, o3, o4)
                if ("gpt-5.1" in model_id or "gpt-4.1" in model_id or "gpt-4o" in model_id or "gpt-4-turbo" in model_id or 
                    (model_id.startswith("gpt-") and not any(r in model_id for r in reasoning_indicators))):
                    available_models = [model.id]
                    log(f"Using available OpenAI model with reasoning_effort=none: {model.id}")
                    break
    except Exception as e:
        log_warn(f"Error checking OpenAI models, using fallback: {e}")
        # Fallback to most likely available non-reasoning model
        available_models = [preferred_models[0]]
    
    if not available_models:
        available_models = [preferred_models[0]]  # Final fallback to GPT-5.1
        log(f"Using fallback GPT-5.1 with reasoning_effort=none: {available_models[0]}")
    
    return available_models

def discover_gemini_models() -> List[str]:
    """Return Gemini 3 Pro for non-reasoning tasks."""
    # Use Gemini 3 Pro - most powerful non-reasoning Gemini model
    preferred_models = [
        "gemini-3-pro",           # Gemini 3 Pro (primary, non-flash)
        "gemini-3-pro-preview",   # Fallback to Gemini 3 Pro preview
        "gemini-2.5-pro",         # Fallback to Gemini 2.5 Pro
        "gemini-2.0-flash",       # Fallback to Gemini 2.0 Flash
    ]
    
    # Exclude reasoning models, embeddings, and specialized models
    reasoning_indicators = ["reasoning", "think", "cot"]
    exclude_indicators = ["embed", "image-generation", "vision", "audio", "code", "live", "native-audio", "tts", "computer-use"] + reasoning_indicators
    
    if not GEMINI_CLIENT:
        # Return Gemini 3 Pro as fallback
        model = "gemini-3-pro"
        log(f"Gemini client not initialized, using gemini-3-pro: {model}")
        return [model]
    
    available_models = []
    try:
        # Try to list available models
        model_list = GEMINI_CLIENT.list_models()
        available_model_names = {model.name.lower() for model in model_list}
        
        # Return the first preferred model that's available
        for model in preferred_models:
            # Check if model is available
            for available_name in available_model_names:
                # Use strict matching to avoid substring issues
                # Match exact model name or models/{model} format
                model_matches = (
                    available_name == model.lower() or 
                    available_name.endswith(f"/{model}") or
                    available_name == f"models/{model}"
                )
                
                # Only proceed if we have an exact match
                if not model_matches:
                    continue
                
                # Check it's not a specialized model
                if any(exclude in available_name for exclude in exclude_indicators):
                    continue
                
                # Extract clean model ID
                model_id = available_name.split("/")[-1] if "/" in available_name else available_name
                available_models = [model_id]
                log(f"Using gemini-3-pro: {model_id}")
                break
            if available_models:
                break
        
        # If none of preferred found, look for any gemini-3-pro variant
        if not available_models:
            for model in model_list:
                model_name = model.name.lower()
                # Skip embeddings and specialized models
                if any(exclude in model_name for exclude in exclude_indicators):
                    continue
                # Look for gemini-3-pro (non-flash, non-image)
                if "gemini-3-pro" in model_name and "flash" not in model_name and "image" not in model_name:
                    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
                    available_models = [model_id]
                    log(f"Using available gemini-3-pro: {model_id}")
                    break
    except Exception as e:
        log_warn(f"Error checking Gemini models, using fallback: {e}")
        available_models = ["gemini-3-pro"]
    
    if not available_models:
        available_models = ["gemini-3-pro"]  # Final fallback to gemini-3-pro
        log(f"Using fallback gemini-3-pro: {available_models[0]}")
    
    return available_models

def discover_claude_models() -> List[str]:
    """Return Claude Opus 4.1 non-reasoning model."""
    # Claude Opus 4.1 - most powerful non-reasoning Claude model
    # Order of preference: Claude Opus 4.1, then fallbacks
    preferred_models = [
        "claude-opus-4-1",                    # Claude Opus 4.1 alias (primary)
        "claude-opus-4-1-20250805",          # Claude Opus 4.1 specific version
        "claude-3-opus-20240229",             # Fallback to Claude 3 Opus
        "claude-3-5-sonnet-20241022",         # Fallback to Claude 3.5 Sonnet
    ]
    
    # Exclude reasoning models (o1, o3, o4)
    reasoning_indicators = ["o1", "o3", "o4", "reasoning"]
    
    # Use Claude Opus 4.1 - most powerful non-reasoning model
    # Anthropic doesn't have a public model listing API, so we'll use the most likely available
    model = preferred_models[0]  # Start with Claude Opus 4.1 (claude-opus-4-1)
    log(f"Using Claude Opus 4.1 non-reasoning model: {model}")
    return [model]

def discover_grok_models() -> List[str]:
    """Return Grok 4.1 Fast non-reasoning model."""
    # Grok 4.1 Fast - optimized for speed, non-reasoning model
    # Order of preference: Grok 4.1 Fast, then fallbacks
    preferred_models = [
        "grok-4-1-fast-non-reasoning",  # Grok 4.1 Fast non-reasoning (primary)
        "grok-4.1-fast",                 # Alternative naming
        "grok-4-fast",                   # Grok 4 Fast
        "grok-4.1",                      # Grok 4.1 (fallback)
        "grok-2",                        # Fallback to Grok 2
    ]
    
    # Exclude reasoning models
    reasoning_indicators = ["reasoning", "think", "cot"]
    exclude_indicators = ["embed", "code"] + reasoning_indicators
    
    if not XAI_API_KEY:
        # Return Grok 4.1 Fast as fallback
        model = preferred_models[0]  # grok-4-1-fast-non-reasoning
        log(f"Grok API key not available, using Grok 4.1 Fast non-reasoning model: {model}")
        return [model]
    
    available_models = []
    try:
        # x.ai API endpoint for models
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.x.ai/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            available_model_ids = {model.get("id", "").lower() for model in data.get("data", [])}
            
            # Return the first preferred non-reasoning model that's available
            for model in preferred_models:
                model_lower = model.lower()
                if model_lower in available_model_ids:
                    # Double-check it's not a reasoning model
                    if not any(r in model_lower for r in reasoning_indicators):
                        available_models = [model]
                        log(f"Using Grok 4.1 Fast non-reasoning model: {model}")
                        break
    except Exception as e:
        log_warn(f"Error checking Grok models, using fallback: {e}")
    
    if not available_models:
        available_models = [preferred_models[0]]  # Fallback to Grok 4.1 Fast
        log(f"Using fallback Grok 4.1 Fast non-reasoning model: {available_models[0]}")
    
    return available_models

def discover_together_models() -> List[str]:
    """Return Llama 4 Maverick from Together AI."""
    if not TOGETHER_API_KEY:
        # Return the model as fallback
        model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        log(f"Together AI API key not found, using fallback model: {model}")
        return [model]
    
    # Together AI model names for Llama 4 Maverick
    # Reference: https://www.together.ai/models/llama-4-maverick
    preferred_models = [
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # Llama 4 Maverick (primary)
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo",  # Turbo version if available
    ]
    
    # Together AI doesn't require listing models - just use the model name
    # The API will return an error if the model doesn't exist
    model = preferred_models[0]
    log(f"Using Llama 4 Maverick from Together AI: {model}")
    return [model]

def get_all_models() -> List[Tuple[str, str]]:
    """Get all models from all providers as (provider, model) tuples."""
    models = []
    
    # Together AI models (Llama 4 Maverick) - test FIRST
    for model in discover_together_models():
        models.append(("together", model))
    
    # OpenAI models
    for model in discover_openai_models():
        models.append(("openai", model))
    
    # Claude models
    for model in discover_claude_models():
        models.append(("claude", model))
    
    # Grok models
    for model in discover_grok_models():
        models.append(("grok", model))
    
    # Gemini models - test last to avoid rate limit issues affecting other models
    for model in discover_gemini_models():
        models.append(("gemini", model))
    
    log(f"Total models to test: {len(models)}")
    return models

# ────────── API Query Functions ──────────
def ask_openai(model: str, prompt: str) -> str:
    """Query OpenAI API."""
    if not OPENAI_CLIENT:
        return "unknown"
    
    for attempt in range(MAX_RETRIES):
        try:
            # Build request parameters
            request_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "timeout": TIMEOUT_SEC,
            }
            
            # For GPT-5.1, use max_completion_tokens instead of max_tokens and set reasoning_effort
            if "gpt-5.1" in model.lower() or "gpt-5" in model.lower():
                # GPT-5.1 may need more tokens even with reasoning_effort="none"
                # Increase to allow the model to generate a response
                request_params["max_completion_tokens"] = 50
                request_params["reasoning_effort"] = "none"
            else:
                # For other models, use max_tokens
                request_params["max_tokens"] = 10
            
            response = OPENAI_CLIENT.chat.completions.create(**request_params)
            
            # Handle None or empty content - this is the main issue causing "unknown" responses
            content = response.choices[0].message.content
            if content is None:
                log_err(f"OpenAI returned None content for {model} on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
            
            stripped = content.strip()
            if not stripped:
                log_err(f"OpenAI returned empty content for {model} on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
            
            return stripped
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                log_warn(f"OpenAI rate limit, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            log_err(f"OpenAI rate limit exceeded for {model}")
            return "unknown"
        except openai.APITimeoutError:
            log_err(f"OpenAI timeout for {model}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
        except openai.NotFoundError:
            log_err(f"OpenAI model not found: {model}")
            return "unknown"
        except openai.BadRequestError as e:
            # Handle invalid parameter errors (e.g., model doesn't exist or parameter not supported)
            error_msg = str(e).lower()
            log_err(f"OpenAI bad request for {model}: {e}")
            # If it's a model not found or invalid parameter, don't retry
            if "model" in error_msg and ("not found" in error_msg or "invalid" in error_msg or "does not exist" in error_msg):
                return "unknown"
            # If it's a parameter error, try without the problematic parameter
            if "max_completion_tokens" in error_msg or "reasoning_effort" in error_msg:
                log_warn(f"Parameter error for {model}, may need to use different parameters")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            # Log the full error for debugging
            log_err(f"OpenAI error for {model} (attempt {attempt + 1}): {error_type}: {e}")
            # Handle network errors
            if "connection" in error_str or "network" in error_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"OpenAI network error, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
            # For other errors, retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
    
    return "unknown"

def ask_gemini(model: str, prompt: str) -> str:
    """Query Google Gemini API."""
    if not GEMINI_CLIENT:
        return "unknown"
    
    # Add small delay to throttle requests and prevent overwhelming API
    time.sleep(0.5)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Use the model name directly or construct full path
            model_name = f"models/{model}" if not model.startswith("models/") else model
            gen_model = GEMINI_CLIENT.GenerativeModel(model_name)
            
            # Build generation config
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 50,  # Increased to ensure response
            }
            
            # For Gemini 3 models, use REST API directly to pass thinking_level
            # The python SDK doesn't support thinking_level in GenerationConfig yet
            if "gemini-3" in model.lower() and "flash" not in model.lower():
                # Use REST API directly for Gemini 3 to support thinking_level
                import json
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    return "unknown"
                
                url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                }
                
                # FIX 1: Increase maxOutputTokens to 200 when using thinkingConfig
                # thinkingConfig uses ~100 tokens for "thoughts" even at "low" level
                # Need 200+ tokens to ensure room for both thinking and output
                max_tokens = 200 if attempt == 0 else 300  # Increase on retry
                
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": max_tokens,
                        "thinkingConfig": {
                            "thinking_level": "low"  # Use thinkingConfig.thinking_level for Gemini 3
                        }
                    }
                }
                
                try:
                    response = requests.post(
                        f"{url}?key={api_key}",
                        headers=headers,
                        json=payload,
                        timeout=TIMEOUT_SEC
                    )
                    
                    # FIX 4: Add comprehensive logging
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Log token usage for debugging
                        usage = result.get("usageMetadata", {})
                        if usage:
                            thoughts_tokens = usage.get("thoughtsTokenCount", 0)
                            total_tokens = usage.get("totalTokenCount", 0)
                            if thoughts_tokens > 0:
                                log(f"Gemini {model} token usage: {thoughts_tokens} thoughts, {total_tokens} total")
                        
                        if "candidates" in result and len(result["candidates"]) > 0:
                            candidate = result["candidates"][0]
                            finish_reason = candidate.get("finishReason")
                            
                            # FIX 2: Handle MAX_TOKENS finishReason explicitly
                            if finish_reason == "MAX_TOKENS":
                                log_warn(f"Gemini {model} MAX_TOKENS - thinking used all tokens, retrying with more tokens")
                                if attempt < MAX_RETRIES - 1:
                                    # Retry with more tokens
                                    time.sleep(RETRY_DELAY)
                                    continue
                                else:
                                    # Last attempt failed, try without thinkingConfig
                                    log_warn(f"Gemini {model} retrying without thinkingConfig")
                                    payload_no_thinking = {
                                        "contents": [{"parts": [{"text": prompt}]}],
                                        "generationConfig": {
                                            "temperature": 0.0,
                                            "maxOutputTokens": 50,
                                        }
                                    }
                                    response = requests.post(
                                        f"{url}?key={api_key}",
                                        headers=headers,
                                        json=payload_no_thinking,
                                        timeout=TIMEOUT_SEC
                                    )
                                    if response.status_code == 200:
                                        result = response.json()
                                        if "candidates" in result and len(result["candidates"]) > 0:
                                            candidate = result["candidates"][0]
                                            finish_reason = candidate.get("finishReason")
                            
                            # Check finish reason
                            if finish_reason == "SAFETY":
                                log_warn(f"Gemini safety filter triggered for {model}")
                                return "unknown"
                            
                            # FIX 2: Check for empty content dict
                            if "content" in candidate:
                                content = candidate["content"]
                                # Handle empty content dict
                                if not content or (isinstance(content, dict) and not content.get("parts")):
                                    log_warn(f"Gemini {model} returned empty content, finishReason={finish_reason}")
                                    if finish_reason == "MAX_TOKENS" and attempt < MAX_RETRIES - 1:
                                        time.sleep(RETRY_DELAY)
                                        continue
                                    return "unknown"
                                
                                # Extract text from parts
                                if "parts" in content and isinstance(content["parts"], list):
                                    for part in content["parts"]:
                                        if isinstance(part, dict) and "text" in part:
                                            text = part["text"].strip()
                                            if text:
                                                return text
                            else:
                                log_warn(f"Gemini {model} no content in candidate, finishReason={finish_reason}")
                        
                        # No valid response found
                        log_warn(f"Gemini {model} no valid response from API")
                        return "unknown"
                        
                    elif response.status_code == 429:
                        # Rate limit - parse Retry-After header if available, but use shorter waits
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                # Use the retry-after value, respect it but add buffer
                                wait_seconds = int(retry_after) + 5  # Respect Retry-After, add buffer
                                log_warn(f"Gemini rate limit (429), Retry-After: {retry_after}s, waiting {wait_seconds}s")
                                if attempt < MAX_RETRIES - 1:
                                    time.sleep(wait_seconds)
                                    continue
                            except (ValueError, TypeError):
                                pass
                        # Fall through to exception handler with rate limit info
                        raise Exception(f"Rate limit 429: {response.text[:200]}")
                    elif response.status_code >= 400:
                        # Error response - log and raise
                        error_text = response.text[:500]  # Limit error text length
                        log_err(f"Gemini REST API error {response.status_code} for {model}: {error_text}")
                        raise Exception(f"API error {response.status_code}: {error_text}")
                    else:
                        log_err(f"Gemini REST API unexpected status {response.status_code} for {model}")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                            continue
                        return "unknown"
                        
                except requests.Timeout:
                    log_err(f"Gemini REST API timeout for {model}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return "unknown"
                except requests.RequestException as e:
                    log_err(f"Gemini REST API request error for {model}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return "unknown"
            else:
                # Use SDK for non-Gemini-3 models
                response = gen_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Handle SDK response object
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    # Check finish reason
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if finish_reason == 2:  # SAFETY - content blocked
                        log_warn(f"Gemini safety filter triggered for {model}")
                        return "unknown"
                    
                    # Try to get text from content.parts
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip()
                
                # Fallback to response.text if available
                if hasattr(response, 'text') and response.text:
                    return response.text.strip()
                
                return "unknown"
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Handle "Unknown field" errors (like thinking_level)
            if "unknown field" in error_str or "Unknown field" in str(e):
                log_err(f"Gemini parameter error for {model}: {e}")
                # Don't retry with same parameters, return unknown
                if attempt == 0 and "thinking" in error_str.lower():
                    log_warn(f"Removing unsupported parameter and retrying {model}")
                    # Could retry without the problematic parameter, but for now just return unknown
                return "unknown"
            
            # Handle rate limits and quotas
            if "rate" in error_str or "quota" in error_str or "429" in str(e) or "ResourceExhausted" in error_type:
                # Use more retries for rate limit errors
                max_retries_for_rate_limit = MAX_RETRIES_RATE_LIMIT if ("429" in str(e) or "rate limit" in error_str) else MAX_RETRIES
                
                if attempt < max_retries_for_rate_limit - 1:
                    # Extract retry delay from error if available
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    
                    # Try to extract wait time from error message
                    if "retry in" in error_str or "retry_delay" in error_str:
                        import re
                        retry_match = re.search(r'retry in ([\d.]+)s', error_str)
                        if retry_match:
                            wait_time = float(retry_match.group(1)) + 5  # Add buffer
                    
                    # For rate limits, use longer exponential backoff to let quota reset
                    if "429" in str(e) or "rate limit" in error_str:
                        wait_time = 30 * (2 ** attempt)  # Start at 30s, double each retry: 30s, 60s, 120s
                    
                    log_warn(f"Gemini rate limit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries_for_rate_limit})")
                    time.sleep(wait_time)
                    continue
                log_err(f"Gemini rate limit exceeded for {model} after {max_retries_for_rate_limit} attempts")
                return "unknown"
            # Handle timeouts
            elif "timeout" in error_str or "timed out" in error_str:
                log_err(f"Gemini timeout for {model}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
            # Handle invalid model errors
            elif "not found" in error_str or "invalid" in error_str or "404" in str(e):
                log_err(f"Gemini model not found or invalid: {model}")
                return "unknown"
            # Handle network errors
            elif "connection" in error_str or "network" in error_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Gemini network error, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                log_err(f"Gemini network error for {model}")
                return "unknown"
            # General error handling
            else:
                log_err(f"Gemini error for {model}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
    
    return "unknown"

def ask_claude(model: str, prompt: str) -> str:
    """Query Anthropic Claude API."""
    if not ANTHROPIC_CLIENT:
        return "unknown"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = ANTHROPIC_CLIENT.messages.create(
                model=model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Handle rate limits
            is_rate_limit = (
                "rate" in error_str or "rate_limit" in error_str or "429" in str(e) or
                (hasattr(anthropic, 'RateLimitError') and isinstance(e, getattr(anthropic, 'RateLimitError', type(None))))
            )
            
            if is_rate_limit:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Claude rate limit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                log_err(f"Claude rate limit exceeded for {model}")
                return "unknown"
            
            # Handle timeouts
            if "timeout" in error_str or "timed out" in error_str:
                log_err(f"Claude timeout for {model}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
            
            # Handle invalid model errors
            if "not found" in error_str or "invalid" in error_str or "404" in str(e):
                log_err(f"Claude model not found or invalid: {model}")
                return "unknown"
            
            # Handle network errors
            if "connection" in error_str or "network" in error_str:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Claude network error, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
            
            # General error handling
            log_err(f"Claude error for {model}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
    
    return "unknown"

def ask_grok(model: str, prompt: str) -> str:
    """Query x.ai Grok API."""
    if not XAI_API_KEY:
        return "unknown"
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            }
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=TIMEOUT_SEC
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Grok rate limit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                log_err(f"Grok rate limit exceeded for {model}")
                return "unknown"
            elif response.status_code == 404:
                log_err(f"Grok model not found: {model}")
                return "unknown"
            elif response.status_code >= 500:
                # Server error, retry
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Grok server error {response.status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                log_err(f"Grok server error for {model}: {response.status_code}")
                return "unknown"
            else:
                log_err(f"Grok error for {model}: {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
        except requests.Timeout:
            log_err(f"Grok timeout for {model}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
        except requests.ConnectionError:
            log_err(f"Grok connection error for {model}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                log_warn(f"Retrying in {wait_time}s")
                time.sleep(wait_time)
                continue
            return "unknown"
        except Exception as e:
            log_err(f"Grok error for {model}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
    
    return "unknown"

def ask_together(model: str, prompt: str) -> str:
    """Query Together AI API for Llama 4 Maverick.
    
    Together AI provides OpenAI-compatible API for Llama 4 Maverick.
    Reference: https://www.together.ai/models/llama-4-maverick
    API Docs: https://docs.together.ai/docs/inference
    """
    if not TOGETHER_API_KEY:
        return "unknown"
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Together AI uses OpenAI-compatible chat completions API
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 50,  # Short response for quiz answers
            }
            
            response = requests.post(
                TOGETHER_API_URL,
                headers=headers,
                json=data,
                timeout=TIMEOUT_SEC
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "").strip()
                    if content:
                        return content
                log_warn(f"Together AI returned empty response for {model}")
                return "unknown"
            elif response.status_code == 429:
                # Rate limit
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Together AI rate limit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                log_err(f"Together AI rate limit exceeded for {model}")
                return "unknown"
            elif response.status_code == 401:
                log_err(f"Together AI authentication error for {model} - check API key")
                return "unknown"
            elif response.status_code == 404:
                log_err(f"Together AI model not found: {model}")
                return "unknown"
            else:
                error_text = response.text[:500] if hasattr(response, 'text') else str(response.status_code)
                log_err(f"Together AI error {response.status_code} for {model}: {error_text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "unknown"
                
        except requests.Timeout:
            log_err(f"Together AI timeout for {model}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
        except requests.ConnectionError:
            log_err(f"Together AI connection error for {model}")
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                log_warn(f"Retrying in {wait_time}s")
                time.sleep(wait_time)
                continue
            return "unknown"
        except Exception as e:
            error_str = str(e).lower()
            log_err(f"Together AI error for {model}: {e}")
            if "rate" in error_str or "429" in str(e):
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    log_warn(f"Together AI rate limit, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return "unknown"
    
    return "unknown"

def ask_api(provider: str, model: str, prompt: str) -> str:
    """Unified wrapper to call appropriate API based on provider."""
    if provider == "together":
        return ask_together(model, prompt)
    elif provider == "openai":
        return ask_openai(model, prompt)
    elif provider == "gemini":
        return ask_gemini(model, prompt)
    elif provider == "claude":
        return ask_claude(model, prompt)
    elif provider == "grok":
        return ask_grok(model, prompt)
    else:
        log_err(f"Unknown provider: {provider}")
        return "unknown"

# ────────── Validation and Scoring (from quiz.py) ──────────
def looks_like_filler_response(text: str) -> bool:
    """Check if response looks like filler text."""
    lowered = text.lower()
    return any(hint in lowered for hint in FILLER_HINTS)

def normalize_and_validate(raw: str) -> Tuple[str, bool, str]:
    """Normalize and validate response, same as quiz.py."""
    if raw is None:
        return "unknown", False, "Empty response"

    stripped = raw.strip()
    if not stripped:
        return "unknown", False, "Empty response"

    normalized = " ".join(stripped.lower().split())
    normalized = ALIAS.get(normalized, normalized)

    if re.fullmatch(r"[1-5](?:[).])?", normalized):
        mapped = NUMERIC_CHOICES[normalized[0]]
        return mapped, True, "Numeric choice"

    if normalized in VALID_ANSWERS and stripped.lower() == normalized:
        return normalized, True, "Exact match"

    stripped_no_punct = stripped.rstrip(".!,?:;")
    collapsed_no_punct = " ".join(stripped_no_punct.lower().split())
    collapsed_no_punct = ALIAS.get(collapsed_no_punct, collapsed_no_punct)

    if collapsed_no_punct in VALID_ANSWERS:
        return collapsed_no_punct, False, "Answer must be exactly one word/phrase with no punctuation."

    for phrase, mapped in FILLER_PHRASES.items():
        if phrase in normalized:
            return mapped, True, f"Alias phrase '{phrase}'"

    word_tokens = [token for token in re.split(r"[^a-z]+", normalized) if token]
    if word_tokens:
        for target, synonyms in FILLER_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in word_tokens:
                    return target, True, f"Alias keyword '{synonym}'"

    for token in VALID_ANSWERS:
        if token in normalized:
            return token, True, "Salvaged token from verbose response"

    return "unknown", False, "Answer must be exactly one of: true, false, possibly true, possibly false, unknown."

def build_reprompt_prompt(base_prompt: str, reason: str, previous: str) -> str:
    """Build reprompt with error feedback."""
    reason_text = reason or "the answer was not in the valid list"
    normalized_previous = " ".join(previous.strip().split()) or "∅"
    normalized_previous = normalized_previous.replace('"', "'")
    return f"{base_prompt}{REPROMPT_TEMPLATE.format(answer=normalized_previous, reason=reason_text)}"

def get_validated_answer(provider: str, model: str, base_prompt: str) -> Tuple[str, str, int, bool, str]:
    """Get validated answer with retries, same logic as quiz.py."""
    attempts = 0
    prompt = base_prompt
    last_raw = ""

    while attempts < MAX_REPROMPTS:
        raw_response = ask_api(provider, model, prompt)
        normalized, valid, reason = normalize_and_validate(raw_response)
        if valid:
            reason_tag = "other"
            if reason:
                if reason.startswith("Alias"):
                    log(f"Alias normalization -> {normalized} | {reason} | raw='{raw_response}'")
                    reason_tag = "alias"
                elif reason == "Numeric choice":
                    log(f"Numeric choice -> {normalized} | raw='{raw_response}'")
                    reason_tag = "numeric"
                elif reason == "Salvaged token from verbose response":
                    log(f"Salvaged verbose -> {normalized} | raw='{raw_response}'")
                    reason_tag = "salvaged_verbose"
                elif reason == "Exact match":
                    log(f"Exact match -> {normalized}")
                    reason_tag = "exact"
                else:
                    log(f"{reason} -> {normalized} | raw='{raw_response}'")
            else:
                reason_tag = "exact"
            return normalized, raw_response, attempts + 1, True, reason_tag

        filler_hit = looks_like_filler_response(raw_response) if raw_response else False
        reprompt_reason = reason
        if filler_hit:
            preview = (raw_response or "").strip().replace("\n", " ")[:80]
            log(f"Filler invalid (provider={provider}, model={model}, attempt={attempts + 1}) -> '{preview}'")
            if attempts == 0:
                reprompt_reason = "your reply began with filler text instead of one of the allowed words"
            else:
                normalized = normalized if normalized in VALID_ANSWERS else "unknown"
                log(f"Filler repeated (provider={provider}, model={model}, attempt={attempts + 1}) -> '{preview}' | aborting")
                return normalized, raw_response, attempts + 1, False, "filler_skipped"

        attempts += 1
        last_raw = raw_response
        if attempts >= MAX_REPROMPTS:
            final_tag = "filler_invalid" if filler_hit else "invalid"
            return normalized, last_raw, attempts, False, final_tag

        prompt = build_reprompt_prompt(base_prompt, reprompt_reason, last_raw)

    return "unknown", last_raw, attempts, False, "invalid"

def build_prompt(field: str, q: str, year: int | None) -> str:
    """Build prompt with instructions, same as quiz.py."""
    yr = f"The paper was published in {year}. " if year else ""
    instructions = (
        "Respond with exactly one lowercase word: true, false, possibly true, possibly false, unknown.\n"
        "Do not add sentences, introductions, or punctuation—phrases like \"here is\" or \"the answer is\" are invalid.\n"
        "Copy the word exactly as spelled above. If you cannot decide, reply unknown. Never leave the reply blank.\n"
    )
    context = f"You are being quizzed in {field}. {yr}"
    return f"{instructions}{context}Question:\n{q}"

def score(gt: str, pred: str) -> float:
    """Score prediction against ground truth, same as quiz.py."""
    if pred == "unknown":
        return 0.0
    if (gt in TRUE_SET and pred in TRUE_SET) or (gt in FALSE_SET and pred in FALSE_SET):
        return CORRECT_SCORE
    return INCORRECT_SCORE

def process_batch(batch: List[Tuple[int, Dict]], provider: str, model: str, field_name: str) -> Tuple[float, int]:
    """Process a batch of questions, same logic as quiz.py."""
    total = 0.0
    salvaged_verbose_count = 0
    full_model_name = f"{provider}:{model}"
    
    for idx, doc in batch:
        q, gt = doc["Question"], doc["Answer"].lower()
        year = doc.get("publication_year")
        prompt = build_prompt(field_name, q, year)

        normalized, raw_answer, attempts_used, valid_answer, reason_tag = get_validated_answer(provider, model, prompt)
        if reason_tag == "salvaged_verbose":
            salvaged_verbose_count += 1
        if not valid_answer:
            if normalized not in VALID_ANSWERS:
                normalized = "unknown"
            filler_notice = reason_tag.startswith("filler_")
            if filler_notice:
                if reason_tag == "filler_skipped":
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {full_model_name} | "
                        f"filler response detected; no retry issued; raw='{raw_answer}' -> using '{normalized}'"
                    )
                else:
                    message = (
                        f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {full_model_name} | "
                        f"filler response ignored after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                    )
                log_err(message)
            else:
                message = (
                    f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {full_model_name} | "
                    f"invalid response after {attempts_used} attempts; raw='{raw_answer}' -> using '{normalized}'"
                )
                if normalized == "unknown" and attempts_used > 1:
                    log_err(message)
                elif normalized == "unknown":
                    log(message)
                else:
                    log(message)

        sc = score(gt, normalized)
        res = "✅" if sc > 0 else "❌" if sc < 0 else "☐"
        attempt_info = f"attempts:{attempts_used}"
        log(
            f"{field_name} | {idx}/{QUESTIONS_PER_FIELD} | {year or '—'} | {full_model_name} | "
            f"GT:{gt} | LLM:{raw_answer} -> {normalized} | {res} | {attempt_info}"
        )
        total += sc
    return total, salvaged_verbose_count

def header() -> List[str]:
    """CSV header, same as quiz.py."""
    return ["model", "overall", *FIELD_MAP.values(), "timestamp"]

def done() -> set[str]:
    """Get set of completed models from CSV."""
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        return {row[0] for row in r}

# ────────── main ──────────
def main():
    """Main evaluation loop, following quiz.py structure."""
    global GLOBAL_SALVAGE_TOTAL, GLOBAL_SALVAGE_BY_MODEL, GLOBAL_SALVAGE_BY_FIELD
    GLOBAL_SALVAGE_TOTAL = 0
    GLOBAL_SALVAGE_BY_MODEL = {}
    GLOBAL_SALVAGE_BY_FIELD = defaultdict(int)
    
    # Initialize API clients
    initialize_api_clients()
    
    # Get all models to test
    all_models = get_all_models()
    if not all_models:
        log_err("No models found to test. Check API keys and installations.")
        return
    
    # Connect to MongoDB
    completed = done()
    hdr = CSV_PATH.exists()
    try:
        mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo.admin.command('ping')
        log("✅ Connected to MongoDB")
    except Exception as e:
        log_err(f"Failed to connect to MongoDB: {e}")
        return
    
    cols = {db: mongo[db]["sources"] for db in DATABASES}
    
    # Process each model
    for provider, model in all_models:
        full_model_name = f"{provider}:{model}"
        if full_model_name in completed:
            log(f"skip {full_model_name}")
            continue
        
        log(f"=== MODEL {full_model_name} ===")
        
        field_scores: Dict[str, float] = {}
        field_salvage_counts: Dict[str, int] = {}
        model_salvage_total = 0
        
        for db in DATABASES:
            fname = FIELD_MAP[db]
            col = cols[db]
            try:
                docs = list(col.aggregate([
                    {"$match": {"Question": {"$ne": None}, "Answer": {"$in": list(VALID_ANSWERS)}}},
                    {"$sample": {"size": QUESTIONS_PER_FIELD}}
                ]))
            except Exception as e:
                log_err(f"Error querying {db}: {e}")
                field_scores[fname] = 0.0
                continue
            
            if not docs:
                log_warn(f"No questions found for {fname}")
                field_scores[fname] = 0.0
                continue
            
            indexed_docs = list(enumerate(docs, 1))
            
            # Process in batches with ThreadPoolExecutor
            total = 0.0
            field_salvaged = 0
            # Reduced concurrency for Gemini to prevent rate limits
            if provider == "gemini":
                max_workers = min(3, len(indexed_docs))  # Reduced to 3 to avoid rate limits
            else:
                max_workers = min(16, len(indexed_docs))  # Reasonable concurrency for other APIs
            
            # Split into batches for parallel processing
            batch_size = max(1, len(indexed_docs) // max_workers)
            batches = []
            for i in range(0, len(indexed_docs), batch_size):
                batches.append(indexed_docs[i:i + batch_size])
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_batch, batch, provider, model, fname)
                    for batch in batches if batch
                ]
                for future in as_completed(futures):
                    try:
                        batch_total, batch_salvaged = future.result()
                        total += batch_total
                        field_salvaged += batch_salvaged
                    except Exception as e:
                        log_err(f"Error processing batch in {fname}: {e}")
            
            model_salvage_total += field_salvaged
            field_salvage_counts[fname] = field_salvaged
            GLOBAL_SALVAGE_BY_FIELD[fname] += field_salvaged
            field_scores[fname] = round(total, 4)
        
        # Calculate overall score and write to CSV
        overall = round(sum(field_scores.values()) / len(field_scores), 4) if field_scores else 0.0
        row = [full_model_name, str(overall), *(str(field_scores.get(f, 0.0)) for f in FIELD_MAP.values()), datetime.now().isoformat()]
        mode = "a" if hdr else "w"
        with CSV_PATH.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not hdr:
                w.writerow(header())
                hdr = True
            w.writerow(row)
        
        GLOBAL_SALVAGE_TOTAL += model_salvage_total
        GLOBAL_SALVAGE_BY_MODEL[full_model_name] = model_salvage_total
        if model_salvage_total:
            per_field = ", ".join(
                f"{field}:{count}"
                for field, count in field_salvage_counts.items()
                if count
            )
            log(f"{full_model_name} salvaged verbose answers: total={model_salvage_total}"
                f"{' | ' + per_field if per_field else ''}")
        log(f"done {full_model_name}")
    
    mongo.close()
    log("🎉 all models done")
    if GLOBAL_SALVAGE_TOTAL:
        log(
            "Verbose answer salvage summary — "
            f"total:{GLOBAL_SALVAGE_TOTAL} | "
            f"models:{ {m:c for m,c in GLOBAL_SALVAGE_BY_MODEL.items() if c} } | "
            f"fields:{ {f:c for f,c in GLOBAL_SALVAGE_BY_FIELD.items() if c} }"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("bye")
    except Exception as e:
        log_err(f"Fatal error: {e}")
        raise

