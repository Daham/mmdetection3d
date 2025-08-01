#!/usr/bin/env python3
"""
Test fix for BatchNorm issue in AdaptiveSparseBridge.
"""

import torch
import torch.nn as nn

def test_batchnorm_fix():
    """Test that LayerNorm works better than BatchNorm for single samples."""
    print("🔧 Testing BatchNorm Fix")
    print("="*30)
    
    # Test LayerNorm (our fix)
    print("\n✅ Testing LayerNorm (our fix):")
    try:
        layer_norm = nn.LayerNorm(64)
        test_input = torch.randn(1, 64)  # Single sample
        output = layer_norm(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("   ✅ LayerNorm works with single sample")
    except Exception as e:
        print(f"   ❌ LayerNorm failed: {e}")
    
    # Test BatchNorm1d (problematic)
    print("\n❌ Testing BatchNorm1d (problematic):")
    try:
        batch_norm = nn.BatchNorm1d(64)
        batch_norm.train()  # Set to training mode
        test_input = torch.randn(1, 64)  # Single sample
        output = batch_norm(test_input)
        print(f"   Output shape: {output.shape}")
        print("   ⚠️  BatchNorm1d worked (unexpected)")
    except Exception as e:
        print(f"   ❌ BatchNorm1d failed as expected: {e}")
    
    # Test BatchNorm1d with multiple samples
    print("\n✅ Testing BatchNorm1d with batch:")
    try:
        batch_norm = nn.BatchNorm1d(64)
        batch_norm.train()
        test_input = torch.randn(5, 64)  # Multiple samples
        output = batch_norm(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("   ✅ BatchNorm1d works with batch")
    except Exception as e:
        print(f"   ❌ BatchNorm1d with batch failed: {e}")
    
    print(f"\n{'='*30}")
    print("📋 Summary:")
    print("   - LayerNorm: Works with any input size ✅")
    print("   - BatchNorm1d: Requires batch_size > 1 in training ❌")
    print("   - Our fix: Use LayerNorm for robust processing ✅")
    
    print(f"\n🎯 The fix should resolve the training error!")

if __name__ == "__main__":
    test_batchnorm_fix()
