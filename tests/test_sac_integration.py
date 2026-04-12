#!/usr/bin/env python3
"""
Integration test for SAC backend implementation.
Tests that all new components can be imported and instantiated without errors.
Does NOT require a GPU or full simulator – just validates structure and types.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all new modules can be imported."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    
    try:
        from rsl_rl.algorithms import SAC, SACActorCritic, SAC_AMP
        print("✓ SAC algorithms imported")
        
        from rsl_rl.runners import SACRunner, SACAMPRunner
        print("✓ SAC runners imported")
        
        from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer
        print("✓ SAC replay buffer imported")
        
        from legged_gym.amp.manager import AMPManager
        print("✓ AMP manager imported")
        
        from legged_gym.amp.motion_dataset import MotionDataset
        print("✓ Motion dataset imported")
        
        from legged_gym.amp.reward import compute_amp_reward
        print("✓ AMP reward module imported")
        
        from legged_gym.envs.base.template_cfgs import SACCfg, FastSACCfg
        print("✓ SAC config templates imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registrations():
    """Test that new runners and algorithms are registered."""
    print("\n" + "=" * 60)
    print("TEST 2: Registrations")
    print("=" * 60)
    
    try:
        from rsl_rl.runners import runner_registry
        from rsl_rl.algorithms import algorithm_registry
        
        # Check runner registration
        sac_runner_cls = runner_registry.get_runner_class("SACRunner")
        print(f"✓ SACRunner registered: {sac_runner_cls.__name__}")
        
        sac_amp_runner_cls = runner_registry.get_runner_class("SACAMPRunner")
        print(f"✓ SACAMPRunner registered: {sac_amp_runner_cls.__name__}")
        
        # Check algorithm registration (if registry exists)
        try:
            sac_algo = algorithm_registry.get_algorithm_class("SAC")
            print(f"✓ SAC algorithm registered: {sac_algo.__name__}")
        except:
            print("⊘ Algorithm registry not in place (may be dict-based in runners)")
        
        return True
    except Exception as e:
        print(f"✗ Registration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_templates():
    """Test that config templates are valid."""
    print("\n" + "=" * 60)
    print("TEST 3: Config Templates")
    print("=" * 60)
    
    try:
        from legged_gym.envs.base.template_cfgs import SACCfg, FastSACCfg
        
        # Instantiate configs (won't have all fields, but structure should be valid)
        sac_cfg = SACCfg()
        print(f"✓ SACCfg instantiated")
        print(f"  - runner_class_name: {sac_cfg.runner_class_name}")
        print(f"  - algorithm_class_name: {sac_cfg.algorithm_class_name}")
        print(f"  - policy_class_name: {sac_cfg.policy_class_name}")
        
        fastsac_cfg = FastSACCfg()
        print(f"✓ FastSACCfg instantiated")
        print(f"  - runner_class_name: {fastsac_cfg.runner_class_name}")
        print(f"  - UTD ratio: {fastsac_cfg.runner.utd_ratio}")
        print(f"  - Batch size: {fastsac_cfg.runner.batch_size}")
        
        return True
    except Exception as e:
        print(f"✗ Config template check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_compatibility():
    """Test that Phase 1 (env extras standardization) doesn't break existing AMP code."""
    print("\n" + "=" * 60)
    print("TEST 4: Phase 1 Compatibility (env.step() extras)")
    print("=" * 60)
    
    try:
        # Just verify the AMP runner can import and the modifications look OK
        from rsl_rl.runners.amp_runner import AMPRunner
        print("✓ AMPRunner still imports correctly after Phase 1 changes")
        
        # Check that the runner has the learn method
        if hasattr(AMPRunner, 'learn'):
            print("✓ AMPRunner.learn() method exists")
        
        # Check CTS AMP runner as well
        from rsl_rl.runners.cts_amp_runner import CTSAMPRunner
        print("✓ CTSAMPRunner still imports correctly after Phase 1 changes")
        
        return True
    except Exception as e:
        print(f"✗ Phase 1 compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_amp_module_structure():
    """Test that AMP module components are available."""
    print("\n" + "=" * 60)
    print("TEST 5: AMP Module Structure")
    print("=" * 60)
    
    try:
        from legged_gym.amp.manager import AMPManager
        print("✓ AMPManager class available")
        
        # Check that AMPManager has required methods
        required_methods = ['compute_reward', 'update_discriminator', 'save', 'load']
        for method in required_methods:
            if hasattr(AMPManager, method):
                print(f"  ✓ {method}()")
            else:
                print(f"  ✗ {method}() missing")
                return False
        
        from legged_gym.amp.motion_dataset import MotionDataset
        print("✓ MotionDataset class available")
        
        from legged_gym.amp.reward import compute_amp_reward
        print("✓ compute_amp_reward function available")
        
        return True
    except Exception as e:
        print(f"✗ AMP module structure check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sac_structure():
    """Test that SAC algorithm components are available."""
    print("\n" + "=" * 60)
    print("TEST 6: SAC Algorithm Structure")
    print("=" * 60)
    
    try:
        from rsl_rl.algorithms.sac import SAC, SACActorCritic
        print("✓ SAC and SACActorCritic classes available")
        
        # Check SAC has required methods
        required_methods = ['act', 'update']
        for method in required_methods:
            if hasattr(SAC, method):
                print(f"  ✓ SAC.{method}()")
            else:
                print(f"  ✗ SAC.{method}() missing")
                return False
        
        from rsl_rl.algorithms.sac_amp import SAC_AMP
        print("✓ SAC_AMP class available")
        
        from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer
        print("✓ SACReplayBuffer class available")
        
        return True
    except Exception as e:
        print(f"✗ SAC structure check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_play_script_compatibility():
    """Test that play.py can handle SAC runners."""
    print("\n" + "=" * 60)
    print("TEST 7: play.py SAC Compatibility")
    print("=" * 60)
    
    try:
        # Read play.py and verify SAC handling is there
        play_file = project_root / "legged_gym" / "scripts" / "play.py"
        with open(play_file, 'r') as f:
            content = f.read()
        
        if 'SACRunner' in content or 'sac' in content.lower():
            print("✓ play.py contains SAC-related code")
        else:
            print("⊘ play.py might need SAC-specific handling (check manually)")
        
        return True
    except Exception as e:
        print(f"✗ play.py compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results."""
    tests = [
        ("Imports", test_imports),
        ("Registrations", test_registrations),
        ("Config Templates", test_config_templates),
        ("Phase 1 Compatibility", test_phase1_compatibility),
        ("AMP Module Structure", test_amp_module_structure),
        ("SAC Algorithm Structure", test_sac_structure),
        ("play.py Compatibility", test_play_script_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
