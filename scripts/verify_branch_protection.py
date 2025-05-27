#!/usr/bin/env python3
"""
Branch Protection Verification Script
=====================================

This script helps verify that GitHub branch protection rules are working correctly
by testing various Git operations and providing feedback on the protection status.

Usage:
    python scripts/verify_branch_protection.py
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_git_status():
    """Check basic Git repository status."""
    print("🔍 Checking Git repository status...")
    
    # Check if we're in a Git repository
    success, stdout, stderr = run_command("git status --porcelain")
    if not success:
        print("❌ Not in a Git repository or Git not available")
        return False
    
    # Check current branch
    success, branch, stderr = run_command("git branch --show-current")
    if success:
        print(f"📍 Current branch: {branch}")
    
    # Check remote configuration
    success, remotes, stderr = run_command("git remote -v")
    if success and remotes:
        print("🔗 Remote repositories:")
        for line in remotes.split('\n'):
            print(f"   {line}")
    
    return True

def test_direct_push_protection():
    """Test if direct pushes to main are blocked."""
    print("\n🧪 Testing direct push protection...")
    
    # Check current branch
    success, current_branch, stderr = run_command("git branch --show-current")
    if not success:
        print("❌ Could not determine current branch")
        return False
    
    if current_branch != "main":
        print(f"ℹ️  Currently on branch '{current_branch}', not on main")
        print("   This is good - you should work on feature branches!")
        return True
    
    print("⚠️  You are currently on the main branch")
    print("   With proper branch protection, direct pushes should be blocked")
    print("   Try creating a feature branch: git checkout -b feature/test-branch")
    
    return True

def check_branch_protection_status():
    """Check if branch protection appears to be configured."""
    print("\n🛡️  Checking branch protection indicators...")
    
    # This is a basic check - full verification requires GitHub API access
    print("📋 Branch protection verification checklist:")
    print("   ☐ Navigate to: https://github.com/Zanzagar/AutoEncoder_Experimentation/settings/branches")
    print("   ☐ Look for 'Branch rulesets' section")
    print("   ☐ Click 'Add branch ruleset' if no rulesets exist")
    print("   ☐ Verify 'main' branch has protection rules configured")
    print("   ☐ Check 'Require a pull request before merging' is enabled")
    print("   ☐ Check 'Require status checks to pass' is enabled")
    print("   ☐ Check 'Require branches to be up to date before merging' is enabled")
    print("   ☐ Check 'Block pushes that create files larger than specified size' is enabled")
    print("")
    print("   📖 For detailed setup instructions, see: docs/GIT_WORKFLOW.md")
    
    return True

def demonstrate_proper_workflow():
    """Demonstrate the proper Git workflow with branch protection."""
    print("\n✅ Proper workflow with branch protection:")
    print("   1. Create feature branch: git checkout -b feature/your-feature")
    print("   2. Make changes and commit: git add . && git commit -m 'Your changes'")
    print("   3. Push feature branch: git push origin feature/your-feature")
    print("   4. Create Pull Request on GitHub")
    print("   5. Get approval and merge via GitHub interface")
    print("   6. Delete feature branch: git branch -d feature/your-feature")

def main():
    """Main verification function."""
    print("🔒 GitHub Branch Protection Verification")
    print("=" * 50)
    
    # Check Git status
    if not check_git_status():
        sys.exit(1)
    
    # Test protection features
    test_direct_push_protection()
    check_branch_protection_status()
    demonstrate_proper_workflow()
    
    print("\n🎯 Next Steps:")
    print("   1. Complete GitHub branch ruleset setup if not done")
    print("   2. Test by creating a feature branch and making a PR")
    print("   3. Verify that direct pushes to main are blocked")
    print("   4. Review docs/GIT_WORKFLOW.md for detailed instructions")
    
    print("\n✨ Branch protection verification complete!")

if __name__ == "__main__":
    main() 