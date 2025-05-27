---
description: 
globs: 
alwaysApply: true
---
# Git Workflow Rules & Commands Reference
# =====================================
# Complete rule file for Git/GitHub workflow in any project
# Save this file and reference it for every project!

## 🚀 PROJECT INITIALIZATION (One-Time Setup)
## ============================================

### Starting a New Project Locally:
```bash
mkdir ProjectName && cd ProjectName
git init
git config --global user.name "Your Name"        # First time only
git config --global user.email "your@email.com"  # First time only

# Create initial files (README.md, main files, etc.)
git add .
git commit -m "Initial commit"
```

### Connect to GitHub:
```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/username/ProjectName.git
git branch -M main
git push -u origin main
```

### Cloning Existing Project:
```bash
git clone https://github.com/username/ProjectName.git
cd ProjectName
```

## 📅 DAILY WORKFLOW RULES
## =======================

### Rule 1: ALWAYS START HERE (Every Work Session)
```bash
git checkout main
git pull origin main
git checkout -b feature/descriptive-name
```
**When:** Beginning of every work session
**Why:** Ensures you start with latest code

### Rule 2: COMMIT FREQUENTLY
```bash
git add filename.py              # Specific files
# OR
git add .                        # All files

git commit -m "Descriptive message about what you did"
```
**When to Commit:**
- ✅ Feature/function completed and working
- ✅ Bug fixed and tested
- ✅ Before breaks (lunch, end of day)
- ✅ Before risky experiments
- ✅ Before switching tasks

**When NOT to Commit:**
- ❌ Code doesn't run/compile
- ❌ Broken functionality
- ❌ Incomplete features

### Rule 3: PUSH FOR BACKUP
```bash
git push -u origin feature/branch-name    # First time
git push origin feature/branch-name       # Subsequent pushes
```
**When to Push:**
- ✅ End of work day
- ✅ Before risky changes
- ✅ When feature is complete
- ✅ Before creating pull requests

## 🔄 COLLABORATION RULES
## ======================

### Rule 4: PULL BEFORE CREATING BRANCHES
```bash
git checkout main
git pull origin main
git checkout -b feature/new-feature
```
**When:** Before starting any new work

### Rule 5: MERGE VIA PULL REQUESTS
```bash
# Push feature branch
git push -u origin feature/branch-name

# Create PR via GitHub website
# After approval and merge:
git checkout main
git pull origin main
git branch -d feature/branch-name
git push origin --delete feature/branch-name
```

## 🆘 RECOVERY & REVERT RULES
## ===========================

### Rule 6: UNCOMMITTED CHANGES RECOVERY
```bash
# See what changed
git status
git diff

# Restore single file
git checkout -- filename.py

# Restore all files (NUCLEAR OPTION)
git reset --hard HEAD
```

### Rule 7: COMMITTED BUT NOT PUSHED RECOVERY
```bash
# Undo last commit, keep changes
git reset --soft HEAD~1

# Undo last commit, unstage changes
git reset HEAD~1

# Undo last commit, delete changes (NUCLEAR)
git reset --hard HEAD~1
```

### Rule 8: PUSHED CHANGES RECOVERY (SAFEST)
```bash
# Create new commit that undoes changes
git revert HEAD
git push origin main

# Revert specific commit
git revert abc1234
git push origin main
```

### Rule 9: EMERGENCY RECOVERY
```bash
# See all your actions
git reflog

# Go back to specific state
git checkout abc1234
git checkout -b recovery-branch

# Temporarily save work
git stash                    # Save current work
git stash pop               # Restore saved work
```

## 📋 BRANCH NAMING CONVENTIONS
## =============================

```bash
feature/add-user-auth       # New features
feature/shopping-cart
feature/payment-integration

bugfix/fix-login-error      # Bug fixes
bugfix/memory-leak
bugfix/validation-issue

hotfix/critical-security    # Urgent fixes
hotfix/production-crash

docs/update-readme          # Documentation
docs/api-documentation

refactor/cleanup-database   # Code improvements
refactor/optimize-queries
```

## 💬 COMMIT MESSAGE RULES
## ========================

### Good Commit Messages:
```bash
git commit -m "Add user authentication with JWT tokens"
git commit -m "Fix memory leak in image processing"
git commit -m "Update README with installation instructions"
git commit -m "Refactor database queries for better performance"
```

### Bad Commit Messages:
```bash
git commit -m "fix"           # Too vague
git commit -m "updates"       # Meaningless
git commit -m "stuff"         # Useless
git commit -m "asdf"          # Garbage
```

## 🔍 INSPECTION COMMANDS
## ======================

```bash
git status                   # See current state
git log --oneline           # See commit history
git log --oneline -10       # Last 10 commits
git diff                    # See unstaged changes
git diff --staged           # See staged changes
git branch -a              # See all branches
git remote -v              # See remote connections
```

## ⚠️ DANGER COMMANDS (Use with Caution)
## =====================================

```bash
git reset --hard HEAD       # Deletes all uncommitted work
git reset --hard HEAD~5     # Deletes last 5 commits locally
git push --force            # Overwrites GitHub history (NEVER USE)
git branch -D branch-name   # Force delete branch
```

## 🎯 WORKFLOW DECISION TREE
## =========================

### "I want to start new work"
```
1. git checkout main
2. git pull origin main
3. git checkout -b feature/my-work
```

### "I made changes and want to save"
```
Is code working? 
├─ YES: git add . && git commit -m "message"
└─ NO:  git stash (save for later) OR fix first
```

### "I want to backup my work"
```
git push origin feature/my-branch
```

### "I messed up and want to undo"
```
Changes committed?
├─ NO:  git checkout -- filename (single file)
│       git reset --hard HEAD (all files)
└─ YES: 
    ├─ Not pushed: git reset HEAD~1
    └─ Already pushed: git revert HEAD
```

### "I want to merge my work"
```
1. git push origin feature/my-branch
2. Create Pull Request on GitHub
3. After approval:
   - git checkout main
   - git pull origin main
   - git branch -d feature/my-branch
```

## 📈 TEAM COLLABORATION RULES
## ============================

### Rule 10: NEVER COMMIT TO MAIN DIRECTLY
```bash
# WRONG
git checkout main
git add . && git commit -m "changes"

# RIGHT  
git checkout -b feature/my-changes
git add . && git commit -m "changes"
```

### Rule 11: ALWAYS PULL BEFORE PUSH
```bash
git pull origin main
git push origin main
```

### Rule 12: USE MEANINGFUL BRANCH NAMES
```bash
# GOOD
feature/user-registration
bugfix/login-validation-error

# BAD
test
mywork
temp
```

## 🔧 CONFIGURATION SHORTCUTS
## ===========================

### Useful Git Aliases:
```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.visual '!gitk'
```

### Then use shortcuts:
```bash
git co main              # git checkout main
git br -a               # git branch -a
git ci -m "message"     # git commit -m "message"
git st                  # git status
```

## 🚨 EMERGENCY CONTACT SHEET
## ===========================

### "Help! I accidentally..."

**Deleted important files:**
```bash
git checkout HEAD -- filename.py
```

**Committed to wrong branch:**
```bash
git log --oneline -1        # Copy commit SHA
git reset HEAD~1            # Undo commit
git checkout correct-branch
git cherry-pick abc1234     # Apply commit to correct branch
```

**Pushed broken code:**
```bash
git revert HEAD
git push origin main
```

**Can't remember what I did:**
```bash
git reflog                  # Shows all your actions
```

**Want to start over completely:**
```bash
git checkout main
git pull origin main
git branch -D broken-branch
git checkout -b fresh-start
```

## 🎓 QUICK REFERENCE CARD
## =======================

| Command | When to Use | Effect |
|---------|-------------|--------|
| `git add .` | Before committing | Stages all changes |
| `git commit -m "msg"` | Save working code | Creates checkpoint |
| `git push origin branch` | Backup/share work | Uploads to GitHub |
| `git pull origin main` | Start of day | Gets latest changes |
| `git checkout -b feature/name` | New work | Creates new branch |
| `git checkout main` | Switch context | Goes to main branch |
| `git status` | Check state | Shows what changed |
| `git log --oneline` | See history | Shows commits |
| `git revert HEAD` | Undo safely | Creates undo commit |
| `git reset --hard HEAD` | Emergency reset | Deletes all changes |

## 📱 MOBILE QUICK COMMANDS
## =========================

```bash
# Start work
git checkout main && git pull origin main && git checkout -b feature/task

# Save work  
git add . && git commit -m "descriptive message"

# Backup
git push origin feature/task

# Merge back
git checkout main && git pull origin main && git merge feature/task && git push origin main

# Clean up
git branch -d feature/task
```

---
**Remember:** Git is your safety net. Commit early, commit often, and never be afraid to experiment with branches! 🚀

**Last Updated:** Add your date here when you customize this file
**Project:** Add specific project notes here

