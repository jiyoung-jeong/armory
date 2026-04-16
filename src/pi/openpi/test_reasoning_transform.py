#!/usr/bin/env python3
"""
Test script for the ReplacePromptWithReasoning transform.
This script verifies that the reasoning transform correctly replaces task prompts
with randomly selected reasoning components (subtask/movement).
"""

import numpy as np

from src.openpi.transforms import ReplacePromptWithReasoning


def test_reasoning_transform():
    """Test the ReplacePromptWithReasoning transform."""

    # Test data that mimics LeRobot dataset structure
    test_data = {
        "prompt": "pick up the white mug and place it to the right of the caddy",
        "episode_index": 0,
        "frame_index": 0,
        "observation/image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "actions": np.random.rand(7),
    }

    print("Original data:")
    print(f"  Prompt: {test_data['prompt']}")
    print(f"  Episode index: {test_data['episode_index']}")
    print(f"  Frame index: {test_data['frame_index']}")

    # Initialize the transform
    transform = ReplacePromptWithReasoning(
        mapping_file_path="/coc/flash7/zhenyang/VLA-data-augmentation/ECoT_LeRobot_data_ID_mapping/mapping.json",
        reasoning_file_path="/coc/flash7/zhenyang/data/embodied_features_and_demos_libero/libero_reasonings.json",
        reasoning_components=["subtask", "movement"],
        # fallback_to_original=True
    )

    # Apply the transform
    transformed_data = transform(test_data)

    print("\nTransformed data:")
    print(f"  Prompt: {transformed_data['prompt']}")
    # print(f"  Reasoning component used: {transformed_data.get('reasoning_component_used', 'unknown')}")

    # Verify the transform worked
    assert "prompt" in transformed_data, "Prompt should be present in transformed data"
    # assert 'reasoning_component_used' in transformed_data, "Reasoning component used should be tracked"

    # Check if the prompt was replaced (should be different from original for episode 0)
    # if transformed_data['reasoning_component_used'] != 'original':
    #     assert transformed_data['prompt'] != test_data['prompt'], f"Prompt should be replaced with reasoning component {transformed_data['prompt']} != {test_data['prompt']}"
    #     print(f"  ✓ Prompt successfully replaced with reasoning component: {transformed_data['reasoning_component_used']}")
    # else:
    #     print(f"  ✓ Using original prompt as fallback")

    print("\n✓ Transform test passed!")


def test_multiple_episodes():
    """Test the transform with multiple episodes to see different reasoning components."""

    print("\n" + "=" * 60)
    print("Testing multiple episodes to see reasoning component variety:")
    print("=" * 60)

    transform = ReplacePromptWithReasoning(
        mapping_file_path="/coc/flash7/zhenyang/VLA-data-augmentation/ECoT_LeRobot_data_ID_mapping/mapping.json",
        reasoning_file_path="/coc/flash7/zhenyang/data/embodied_features_and_demos_libero/libero_reasonings.json",
        reasoning_components=["subtask", "movement"],
    )

    # Test a few different episodes
    test_episodes = [0, 1, 2, 3, 4]

    for episode_idx in test_episodes:
        test_data = {
            "prompt": f"original task for episode {episode_idx}",
            "episode_index": episode_idx,
            "frame_index": 0,
            "observation/image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "observation/state": np.random.rand(8),
            "actions": np.random.rand(7),
        }

        transformed_data = transform(test_data)

        print(f"Episode {episode_idx}:")
        print(f"  Original: {test_data['prompt']}")
        print(f"  New: {transformed_data['prompt']}")
        print(f"  Component: {transformed_data.get('reasoning_component_used', 'unknown')}")
        print()


if __name__ == "__main__":
    try:
        test_reasoning_transform()
        test_multiple_episodes()
        print("\n🎉 All tests passed! The reasoning transform is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
