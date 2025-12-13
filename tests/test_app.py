
import unittest
import sys
import os
import json
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock services to prevent startup hangs
sys.modules['rag_engine'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['llm_interface'] = MagicMock()

from app import app

class KiroTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        app.secret_key = 'test_secret'

    def test_homepage(self):
        """Test homepage loads (redirect to login)"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 302)

    def test_guest_login(self):
        """Test guest login"""
        response = self.app.get('/guest_login', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Start Health Assessment', response.data)

if __name__ == '__main__':
    unittest.main()
