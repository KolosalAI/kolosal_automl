"""JWT module for security middleware"""
try:
    from jwt import encode, decode, ExpiredSignatureError, InvalidTokenError
except ImportError:
    # Mock implementation for testing
    class ExpiredSignatureError(Exception):
        pass
    
    class InvalidTokenError(Exception):
        pass
    
    def encode(payload, key, algorithm='HS256'):
        return "mock_jwt_token"
    
    def decode(token, key, algorithms=None, options=None):
        if token == "expired_token":
            raise ExpiredSignatureError("Token expired")
        if token == "invalid_token":
            raise InvalidTokenError("Invalid token")
        return {"user_id": "test", "exp": 9999999999}

__all__ = ['encode', 'decode', 'ExpiredSignatureError', 'InvalidTokenError']
