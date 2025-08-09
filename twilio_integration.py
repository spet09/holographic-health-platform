"""
Twilio SMS integration
"""

import os
from twilio.rest import Client
from dotenv import load_dotenv
from typing import Dict, Any
import logging

load_dotenv()


class TwilioSMSManager:
    """Handle Twilio SMS communications"""

    def __init__(self):
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_phone = os.getenv("TWILIO_PHONE_NUMBER")

        if not all([account_sid, auth_token, self.from_phone]):
            raise ValueError("Missing Twilio configuration")

        self.client = Client(account_sid, auth_token)

    def send_health_alert(self, to_phone: str, life_force_percentage: float,
                          urgent: bool = False) -> Dict[str, Any]:
        """Send health alert SMS"""
        try:
            if urgent:
                message_body = f"""
🚨 URGENT HEALTH ALERT 🚨

Your holographic analysis shows critically low life force: {life_force_percentage:.1f}%

Please seek immediate medical attention.

This is an automated alert from Holographic Health Platform.
"""
            else:
                message_body = f"""
📊 Health Analysis Complete

Life Force Level: {life_force_percentage:.1f}%

View your full report at: {os.getenv('NGROK_URL', 'your-platform-url.com')}

- Holographic Health Platform
"""

            message = self.client.messages.create(
                body=message_body,
                from_=self.from_phone,
                to=to_phone
            )

            return {
                'success': True,
                'message_sid': message.sid,
                'status': message.status
            }

        except Exception as e:
            logging.error(f"Twilio SMS error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def send_custom_message(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send custom SMS message"""
        try:
            sms = self.client.messages.create(
                body=f"{message}\n\n- Holographic Health Platform",
                from_=self.from_phone,
                to=to_phone
            )

            return {
                'success': True,
                'message_sid': sms.sid
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }