"""
Stripe payment processing integration
"""

import os
import stripe
from dotenv import load_dotenv
from typing import Dict, Any
import logging

load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


class StripePaymentProcessor:
    """Handle Stripe payment processing"""

    def __init__(self):
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        self.publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY")

    def create_payment_intent(self, amount: int, currency: str = "usd",
                              customer_email: str = None) -> Dict[str, Any]:
        """Create a payment intent for subscription"""
        try:
            payment_intent = stripe.PaymentIntent.create(
                amount=amount,  # Amount in cents
                currency=currency,
                automatic_payment_methods={
                    'enabled': True,
                },
                receipt_email=customer_email,
                metadata={
                    'source': 'holographic_health_platform'
                }
            )

            return {
                'success': True,
                'client_secret': payment_intent.client_secret,
                'payment_intent_id': payment_intent.id
            }

        except stripe.error.StripeError as e:
            logging.error(f"Stripe error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_customer(self, email: str, name: str = None) -> Dict[str, Any]:
        """Create a Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    'source': 'holographic_health_platform'
                }
            )

            return {
                'success': True,
                'customer_id': customer.id
            }

        except stripe.error.StripeError as e:
            return {
                'success': False,
                'error': str(e)
            }

    def create_subscription(self, customer_id: str, price_id: str) -> Dict[str, Any]:
        """Create a subscription for recurring payments"""
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': price_id}],
                metadata={
                    'source': 'holographic_health_platform'
                }
            )

            return {
                'success': True,
                'subscription_id': subscription.id,
                'status': subscription.status
            }

        except stripe.error.StripeError as e:
            return {
                'success': False,
                'error': str(e)
            }

    def handle_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )

            # Handle different event types
            if event['type'] == 'payment_intent.succeeded':
                payment_intent = event['data']['object']
                # Update database with successful payment

            elif event['type'] == 'invoice.payment_succeeded':
                invoice = event['data']['object']
                # Handle successful subscription payment

            elif event['type'] == 'customer.subscription.deleted':
                subscription = event['data']['object']
                # Handle subscription cancellation

            return {'success': True, 'event_type': event['type']}

        except ValueError as e:
            return {'success': False, 'error': 'Invalid payload'}
        except stripe.error.SignatureVerificationError as e:
            return {'success': False, 'error': 'Invalid signature'}