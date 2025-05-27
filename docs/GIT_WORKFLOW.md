# Git Workflow and Branch Protection Guide

## 🔒 Branch Protection Setup

### GitHub Branch Ruleset Configuration

To ensure code quality and prevent accidental changes to the main branch, follow these steps to configure branch protection using GitHub's current Branch Rulesets:

#### 1. Access Branch Protection Settings
1. Navigate to: https://github.com/Zanzagar/AutoEncoder_Experimentation
2. Click **"Settings"** tab
3. In the left sidebar, click **"Branches"**

#### 2. Add Branch Ruleset for Main Branch
1. Click **"Add branch ruleset"** button
2. Configure the ruleset:
   - **Name**: Enter a descriptive name like "Main Branch Protection"
   - **Enforcement status**: Select "Active"
   - **Bypass list**: Leave empty (or add yourself if you need admin bypass during setup)

#### 3. Configure Target Branches
In the **"Target branches"** section:
1. Click **"Add target"**
2. Select **"Default"** (this targets your main branch automatically)

#### 4. Configure Branch Rules

**✅ Essential Rules (Recommended Settings):**

**Core Protection:**
- ☑️ **"Require a pull request before merging"** ⭐ **MOST IMPORTANT**
  - Click **"Show additional settings"** to configure:
    - Set **"Required number of approvals"** to: `1`
    - Consider enabling **"Dismiss stale pull request approvals when new commits are pushed"**

**Branch Protection:**
- ☑️ **"Restrict deletions"** - **Recommended**
  - Prevents accidental deletion of the main branch

- ☑️ **"Block force pushes"** - **Recommended**
  - Prevents force pushes that could rewrite history

**Additional Security:**
- ☑️ **"Restrict updates"** - **Optional but useful**
  - Only allows users with bypass permission to update matching refs

**🔧 Optional Rules (can enable later):**
- ☑️ **"Require linear history"** - Forces rebase workflow, prevents merge commits
- ☑️ **"Require signed commits"** - Requires GPG-signed commits
- ☑️ **"Require code scanning results"** - Can add when code scanning is set up

**❌ Skip These for Now:**
- ⬜ **"Require status checks to pass"** - ⚠️ **Skip until CI/CD is set up** (requires specific status checks)
- ⬜ **"Restrict creations"** - Not needed for main branch protection
- ⬜ **"Require deployments to succeed"** - Not applicable without deployment setup

#### 5. Save Configuration
1. Review your settings
2. Click **"Create"** button to save the branch ruleset

### 🎯 Minimum Essential Protection

**Core protection for the AutoEncoder project:**

1. **"Require a pull request before merging"** - Prevents direct pushes to main
2. **"Restrict deletions"** - Prevents accidental branch deletion
3. **"Block force pushes"** - Prevents history rewriting

These three rules provide excellent protection for your main branch without requiring CI/CD setup.

**Note:** We'll add "Require status checks to pass" later when we implement automated testing and CI/CD pipelines.

### Alternative: Classic Branch Protection Rules

If you prefer the classic interface or don't see Branch Rulesets:
1. Look for **"Branch protection rules"** section on the same page
2. Click **"Add rule"** (if available)
3. Follow the original process with branch name pattern: `main`

## 🔄 Development Workflow

### Daily Development Process

#### 1. Start New Work
```bash
# Always start from main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/descriptive-name
```

#### 2. Make Changes
```bash
# Make your code changes
# Edit files, add features, fix bugs

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: descriptive commit message"
```

#### 3. Push and Create Pull Request
```bash
# Push feature branch
git push -u origin feature/descriptive-name

# Create Pull Request via GitHub web interface
# Navigate to repository and click "Compare & pull request"
```

#### 4. Code Review and Merge
1. Request review from team members
2. Address any feedback
3. Ensure all status checks pass
4. Merge via GitHub interface (not command line)

#### 5. Cleanup
```bash
# After merge, switch back to main and clean up
git checkout main
git pull origin main
git branch -d feature/descriptive-name
git push origin --delete feature/descriptive-name
```

## 📋 Branch Naming Conventions

### Feature Branches
- `feature/migrate-data-module` - New features or major changes
- `feature/add-visualization-wrapper` - Specific feature additions
- `feature/create-autoencoder-interface` - Interface implementations

### Bug Fix Branches
- `bugfix/visualization-consistency` - Bug fixes
- `bugfix/seed-reproducibility` - Specific bug fixes
- `bugfix/memory-leak-training` - Critical bug fixes

### Documentation Branches
- `docs/update-readme` - Documentation updates
- `docs/api-documentation` - API documentation
- `docs/setup-instructions` - Setup and installation docs

### Hotfix Branches (Emergency)
- `hotfix/critical-security-fix` - Critical production fixes
- `hotfix/data-corruption-fix` - Emergency data fixes

## ✅ Commit Message Guidelines

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```bash
# Good commit messages
git commit -m "feat(data): add geological dataset generation with visualization"
git commit -m "fix(models): resolve memory leak in autoencoder training loop"
git commit -m "docs(readme): update installation instructions for Windows"
git commit -m "refactor(visualization): extract t-SNE plotting into separate module"

# Bad commit messages (avoid these)
git commit -m "fix stuff"
git commit -m "updates"
git commit -m "working version"
```

## 🧪 Testing Branch Protection

### Verification Script
Run the verification script to check branch protection status:

```bash
python scripts/verify_branch_protection.py
```

### Manual Testing
1. Try to push directly to main (should be blocked):
   ```bash
   # This should fail with branch protection
   git checkout main
   echo "test" >> test.txt
   git add test.txt
   git commit -m "test commit"
   git push origin main  # Should be blocked
   ```

2. Test proper workflow:
   ```bash
   # This should work
   git checkout -b feature/test-protection
   echo "test" >> test.txt
   git add test.txt
   git commit -m "test commit"
   git push origin feature/test-protection  # Should succeed
   # Then create PR via GitHub interface
   ```

## 🚨 Emergency Procedures

### Hotfix Process
For critical issues that need immediate attention:

1. **Create hotfix branch from main:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b hotfix/critical-issue-description
   ```

2. **Make minimal fix:**
   ```bash
   # Make only the necessary changes
   git add .
   git commit -m "hotfix: fix critical issue description"
   git push origin hotfix/critical-issue-description
   ```

3. **Create emergency PR:**
   - Mark PR as "urgent" or "hotfix"
   - Request immediate review
   - Merge as soon as approved

### Rollback Procedures
If a merge causes issues:

1. **Identify problematic commit:**
   ```bash
   git log --oneline
   ```

2. **Create revert:**
   ```bash
   git checkout main
   git pull origin main
   git revert <commit-hash>
   git push origin main
   ```

## 📊 Workflow Monitoring

### Status Checks
- All commits must pass automated tests
- Code formatting checks (black, flake8)
- Documentation builds successfully
- No merge conflicts

### Review Requirements
- At least 1 approval required
- All conversations must be resolved
- Branch must be up to date with main

## 🔧 Troubleshooting

### Common Issues

#### "Push to main blocked"
**Solution:** This is expected! Create a feature branch:
```bash
git checkout -b feature/your-changes
git push origin feature/your-changes
```

#### "Branch not up to date"
**Solution:** Update your branch:
```bash
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
```

#### "Status checks failing"
**Solution:** Fix the issues and push again:
```bash
# Fix code formatting
black .
flake8 .

# Fix any test failures
pytest

# Commit fixes
git add .
git commit -m "fix: resolve status check issues"
git push origin your-feature-branch
```

## 📞 Support

For questions about the Git workflow:
1. Check this documentation first
2. Run the verification script: `python scripts/verify_branch_protection.py`
3. Open an issue on GitHub with the "workflow" label
4. Contact the development team

---

**Last Updated:** 2025-05-27  
**Version:** 1.3  
**Status:** ✅ Active - Updated with exact GitHub Branch Rulesets interface options 