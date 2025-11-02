#!/usr/bin/env python3
"""Mochi flashcard management script.

Usage:
    python list_mochi_cards.py          # List all cards in AI/ML deck
    python list_mochi_cards.py test     # Test create/update/delete operations

API Functions:
    get_decks()                         # List all decks
    get_cards(deck_id, limit=100)       # Get all cards in a deck
    create_card(deck_id, content, **kwargs)  # Create a new card
    update_card(card_id, **kwargs)      # Update an existing card
    delete_card(card_id)                # Delete a card

Example:
    from list_mochi_cards import create_card, update_card, delete_card

    # Create a card
    card = create_card(deck_id, "What is X?\n---\nX is Y")

    # Update a card
    update_card(card['id'], content="Updated content")

    # Delete a card
    delete_card(card['id'])
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MOCHI_API_KEY")

if not API_KEY:
    raise ValueError("MOCHI_API_KEY not found in .env file")

BASE_URL = "https://app.mochi.cards/api"


def get_decks():
    """Fetch all decks."""
    response = requests.get(
        f"{BASE_URL}/decks/",
        auth=(API_KEY, ""),
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
    return data["docs"]


def create_card(deck_id, content, **kwargs):
    """Create a new card.

    Args:
        deck_id: Deck ID to add the card to
        content: Markdown content of the card
        **kwargs: Optional fields like template-id, review-reverse?, pos, manual-tags, fields

    Returns:
        Created card data
    """
    data = {
        "content": content,
        "deck-id": deck_id,
        **kwargs
    }

    response = requests.post(
        f"{BASE_URL}/cards/",
        auth=(API_KEY, ""),
        json=data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def update_card(card_id, **kwargs):
    """Update an existing card.

    Args:
        card_id: ID of the card to update
        **kwargs: Fields to update (content, deck-id, archived?, trashed?, etc.)

    Returns:
        Updated card data
    """
    response = requests.post(
        f"{BASE_URL}/cards/{card_id}",
        auth=(API_KEY, ""),
        json=kwargs,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def delete_card(card_id):
    """Delete a card.

    Args:
        card_id: ID of the card to delete

    Returns:
        True if successful
    """
    response = requests.delete(
        f"{BASE_URL}/cards/{card_id}",
        auth=(API_KEY, ""),
        timeout=30
    )
    response.raise_for_status()
    return True


def get_cards(deck_id, limit=100):
    """Fetch all cards for a given deck."""
    cards = []
    bookmark = None

    while True:
        params = {"deck-id": deck_id, "limit": limit}
        if bookmark:
            params["bookmark"] = bookmark

        try:
            response = requests.get(
                f"{BASE_URL}/cards/",
                auth=(API_KEY, ""),
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            batch_size = len(data["docs"])
            if batch_size == 0:
                # No more cards to fetch
                break

            cards.extend(data["docs"])

            bookmark = data.get("bookmark")
            if not bookmark:
                break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500 and len(cards) > 0:
                # API pagination bug - return what we have
                print(f"Note: API error on pagination, showing {len(cards)} cards retrieved\n")
                break
            raise

    return cards


def test_card_operations(deck_id):
    """Test create, update, and delete operations with a temporary card."""
    print("\n" + "=" * 80)
    print("TESTING CARD OPERATIONS (creating temporary test card)")
    print("=" * 80)

    # Create a test card
    print("\n1. Creating test card...")
    test_content = """What is a **test card**?
---
This is a temporary test card created by the API. It will be deleted shortly."""

    created_card = create_card(deck_id, test_content)
    card_id = created_card["id"]
    print(f"   ✓ Created card with ID: {card_id}")
    print(f"   Content: {test_content.split('---')[0].strip()}")

    # Update the card
    print("\n2. Updating test card...")
    updated_content = """What is an **updated test card**?
---
This card has been updated via the API. It will be deleted shortly."""

    update_card(card_id, content=updated_content)
    print(f"   ✓ Updated card {card_id}")
    print(f"   New content: {updated_content.split('---')[0].strip()}")

    # Delete the card
    print("\n3. Deleting test card...")
    delete_card(card_id)
    print(f"   ✓ Deleted card {card_id}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


def list_cards(deck_id, deck_name):
    """List all cards in a deck."""
    print(f"Found deck: {deck_name}\n")
    print("Fetching cards...")
    cards = get_cards(deck_id)

    print(f"\nTotal cards: {len(cards)}")
    print("=" * 80)

    for i, card in enumerate(cards, 1):
        print(f"\nCard {i}:")
        print(f"ID: {card['id']}")
        content = card.get('content', '')
        if len(content) > 200:
            print(f"Content:\n{content[:200]}...")
        else:
            print(f"Content:\n{content}")
        print("-" * 80)


def main():
    import sys

    # Find AI/ML deck
    print("Fetching decks...")
    decks = get_decks()

    aiml_deck = None
    for deck in decks:
        if "AI/ML" in deck["name"] or "AIML" in deck["name"]:
            aiml_deck = deck
            break

    if not aiml_deck:
        print("\nAvailable decks:")
        for deck in decks:
            print(f"  - {deck['name']} (id: {deck['id']})")
        print("\nNo 'AI/ML' deck found. Please check the deck name.")
        return

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_card_operations(aiml_deck["id"])
    else:
        list_cards(aiml_deck["id"], aiml_deck["name"])


if __name__ == "__main__":
    main()
