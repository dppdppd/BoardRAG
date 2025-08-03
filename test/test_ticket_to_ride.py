"""
Tests for the Ticket to Ride game.
"""

from test.query_and_validate_tests import query_and_validate


def test_ticket_to_ride():
    assert query_and_validate(
        question="How many train cards does each player start with in Ticket to Ride? (Answer with the number only)",
        expected_response="4",
    )
