#!/usr/bin/env python3
"""Mochi flashcard management script.

Usage:
    python list_mochi_cards.py          # List all cards in AI/ML deck
    python list_mochi_cards.py test     # Test create/update/delete operations
    python list_mochi_cards.py grade    # Grade all cards using LLM (shows cards < 10)

API Functions:
    get_decks()                         # List all decks
    get_cards(deck_id, limit=100)       # Get all cards in a deck
    create_card(deck_id, content, **kwargs)  # Create a new card
    update_card(card_id, **kwargs)      # Update an existing card
    delete_card(card_id)                # Delete a card
    grade_all_cards(deck_id, batch_size=20)  # Grade cards using LLM

Example:
    from list_mochi_cards import create_card, update_card, delete_card, grade_all_cards

    # Create a card
    card = create_card(deck_id, "What is X?\n---\nX is Y")

    # Update a card
    update_card(card['id'], content="Updated content")

    # Delete a card
    delete_card(card['id'])

    # Grade cards
    imperfect_cards, all_results = grade_all_cards(deck_id)
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MOCHI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("MOCHI_API_KEY not found in .env file")

BASE_URL = "https://app.mochi.cards/api"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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


def grade_cards_batch(cards_batch):
    """Grade a batch of cards using OpenRouter's Gemini 2.5 Flash.

    Args:
        cards_batch: List of card objects to grade

    Returns:
        List of tuples: (card, score, justification)
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Build the prompt with all cards
    prompt = """You are grading flashcards for accuracy. For each card below, evaluate if the answer is correct and complete.

Score each card from 0-10:
- 10: Perfect answer, completely accurate
- 7-9: Mostly correct with minor issues
- 4-6: Partially correct but missing key information
- 0-3: Incorrect or severely incomplete

Format your response as JSON array:
[
  {"card_id": "id1", "score": 10, "justification": "explanation"},
  {"card_id": "id2", "score": 8, "justification": "explanation"}
]

Cards to grade:
"""

    for i, card in enumerate(cards_batch, 1):
        content = card.get('content', '')
        # Split on --- to get question and answer
        parts = content.split('---', 1)
        question = parts[0].strip() if len(parts) > 0 else ''
        answer = parts[1].strip() if len(parts) > 1 else ''

        prompt += f"\n{i}. Card ID: {card['id']}\n"
        prompt += f"   Question: {question}\n"
        prompt += f"   Answer: {answer}\n"

    # Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=data,
        timeout=60
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON response
    import json
    try:
        grades = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response content: {content[:500]}")
        raise

    # Handle both array and object with array
    if isinstance(grades, dict) and 'grades' in grades:
        grades = grades['grades']
    elif isinstance(grades, dict) and 'cards' in grades:
        grades = grades['cards']
    elif isinstance(grades, list):
        pass  # Already a list
    else:
        # Try to extract array from any key
        for key in grades.keys():
            if isinstance(grades[key], list):
                grades = grades[key]
                break

    # Match grades with cards
    results = []
    grade_map = {g['card_id']: (g['score'], g['justification']) for g in grades}

    for card in cards_batch:
        card_id = card['id']
        if card_id in grade_map:
            score, justification = grade_map[card_id]
            results.append((card, score, justification))
        else:
            # Card wasn't graded - add warning
            print(f"Warning: Card {card_id} was not graded by the LLM")

    return results


def grade_all_cards(deck_id, batch_size=20):
    """Grade all cards in a deck, batching requests to minimize API calls.

    Args:
        deck_id: Deck ID to grade cards from
        batch_size: Number of cards per API request (default: 20)

    Returns:
        List of tuples: (card, score, justification) for cards scoring < 10
    """
    print("\nFetching cards to grade...")
    cards = get_cards(deck_id)
    total_cards = len(cards)

    print(f"Grading {total_cards} cards in batches of {batch_size}...")

    all_results = []
    for i in range(0, total_cards, batch_size):
        batch = cards[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_cards + batch_size - 1) // batch_size

        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} cards)...", flush=True)

        try:
            results = grade_cards_batch(batch)
            all_results.extend(results)
        except Exception as e:
            print(f"  Error grading batch {batch_num}: {e}")
            continue

    # Filter cards with score < 10
    imperfect_cards = [(card, score, justification)
                       for card, score, justification in all_results
                       if score < 10]

    return imperfect_cards, all_results


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


def display_grading_results(imperfect_cards, all_results):
    """Display grading results."""
    print("\n" + "=" * 80)
    print("GRADING RESULTS")
    print("=" * 80)

    total_graded = len(all_results)
    perfect_count = total_graded - len(imperfect_cards)

    print(f"\nTotal cards graded: {total_graded}")
    print(f"Perfect scores (10/10): {perfect_count}")
    print(f"Cards needing review (< 10): {len(imperfect_cards)}")

    if imperfect_cards:
        print("\n" + "=" * 80)
        print("CARDS NEEDING REVIEW")
        print("=" * 80)

        # Sort by score (lowest first)
        imperfect_cards.sort(key=lambda x: x[1])

        for i, (card, score, justification) in enumerate(imperfect_cards, 1):
            content = card.get('content', '')
            parts = content.split('---', 1)
            question = parts[0].strip() if len(parts) > 0 else ''
            answer = parts[1].strip() if len(parts) > 1 else ''

            print(f"\n{i}. Score: {score}/10")
            print(f"   Card ID: {card['id']}")
            print(f"   Question: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"   Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            print(f"   Issue: {justification}")
            print("-" * 80)
    else:
        print("\n🎉 All cards are perfect!")


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
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            test_card_operations(aiml_deck["id"])
        elif command == "grade":
            imperfect_cards, all_results = grade_all_cards(aiml_deck["id"])
            display_grading_results(imperfect_cards, all_results)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python list_mochi_cards.py [list|test|grade]")
    else:
        list_cards(aiml_deck["id"], aiml_deck["name"])


if __name__ == "__main__":
    main()
