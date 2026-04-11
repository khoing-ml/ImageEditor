#!/usr/bin/env python3
"""
Auto Git Commit and Push Tool
Streamlines git workflow with optional conventional commit format

Usage:
    python scripts/auto_git.py "feature: add new feature"
    python scripts/auto_git.py -i  # interactive mode
    python scripts/auto_git.py --conventional  # with conventional commit format
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import re


class Colors:
    """ANSI color codes"""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    RESET = '\033[0m'


class GitAutoCommit:
    """Handles automated git commit and push operations"""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize with optional repo root path"""
        self.repo_root = repo_root or Path.cwd()
        os.chdir(self.repo_root)
        self.branch = None

    def log(self, message: str, color: str = Colors.RESET):
        """Print colored log message"""
        print(f"{color}{message}{Colors.RESET}")

    def run_command(self, cmd: list, capture: bool = False) -> str:
        """Run shell command and return output"""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=capture,
                text=True
            )
            return result.stdout.strip() if capture else ""
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Error: {e.stderr}", Colors.RED)
            sys.exit(1)

    def check_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        return (self.repo_root / '.git').exists()

    def get_status(self) -> str:
        """Get git status"""
        return self.run_command(['git', 'status', '--porcelain'], capture=True)

    def get_branch(self) -> str:
        """Get current git branch"""
        if not self.branch:
            self.branch = self.run_command(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture=True
            )
        return self.branch

    def has_changes(self) -> bool:
        """Check if there are unstaged/uncommitted changes"""
        return bool(self.get_status())

    def validate_commit_message(self, message: str) -> bool:
        """Validate commit message"""
        if not message or not message.strip():
            self.log("❌ Commit message cannot be empty", Colors.RED)
            return False
        
        if len(message) > 100:
            self.log("⚠️  Warning: Commit message is longer than 100 characters", Colors.YELLOW)
        
        return True

    def validate_conventional_commit(self, message: str) -> bool:
        """Validate conventional commit format"""
        # Pattern: type(scope)?: description
        pattern = r'^(feat|fix|docs|style|refactor|perf|test|chore|ci)(\(.+?\))?: .+'
        if not re.match(pattern, message):
            self.log(
                "❌ Invalid conventional commit format\n"
                "   Format: <type>[optional scope]: <description>\n"
                "   Types: feat, fix, docs, style, refactor, perf, test, chore, ci",
                Colors.RED
            )
            return False
        return True

    def prompt_for_message(self) -> str:
        """Prompt user for commit message"""
        print(f"{Colors.YELLOW}📝 Enter commit message:{Colors.RESET}")
        message = input("> ").strip()
        return message

    def prompt_for_conventional(self) -> str:
        """Prompt user for conventional commit message"""
        print(f"{Colors.YELLOW}📝 Conventional Commit (type(scope): description):{Colors.RESET}")
        print("Commit types: feat, fix, docs, style, refactor, perf, test, chore, ci")
        
        commit_type = input("Type: ").strip().lower()
        scope = input("Scope (optional, press Enter to skip): ").strip()
        description = input("Description: ").strip()
        
        message = f"{commit_type}"
        if scope:
            message += f"({scope})"
        message += f": {description}"
        
        return message

    def stage_changes(self) -> None:
        """Stage all changes"""
        self.log("📦 Staging changes...", Colors.YELLOW)
        self.run_command(['git', 'add', '-A'])
        self.log("✅ Changes staged", Colors.GREEN)

    def commit(self, message: str) -> None:
        """Create commit"""
        self.log(f"💾 Committing: \"{message}\"", Colors.YELLOW)
        self.run_command(['git', 'commit', '-m', message])
        self.log("✅ Commit successful", Colors.GREEN)

    def push(self) -> None:
        """Push to remote"""
        branch = self.get_branch()
        self.log(f"🌿 Current branch: {branch}", Colors.YELLOW)
        self.log(f"🚀 Pushing to origin/{branch}...", Colors.YELLOW)
        self.run_command(['git', 'push', 'origin', branch])
        self.log("✅ Push successful", Colors.GREEN)

    def show_status(self) -> None:
        """Show current changes"""
        status = self.get_status()
        if status:
            print(f"{Colors.YELLOW}📊 Changes detected:{Colors.RESET}")
            print(status)
        else:
            print(f"{Colors.YELLOW}⚠️  No changes to commit{Colors.RESET}")

    def run(self, message: Optional[str] = None, interactive: bool = False, 
            conventional: bool = False) -> None:
        """Run the full commit and push workflow"""
        
        # Validate repo
        if not self.check_git_repo():
            self.log("❌ Not a git repository", Colors.RED)
            sys.exit(1)

        self.log(f"📁 Project root: {self.repo_root}", Colors.YELLOW)

        # Check for changes
        if not self.has_changes():
            self.log("⚠️  No changes to commit", Colors.YELLOW)
            return

        self.show_status()

        # Get commit message
        if interactive or not message:
            if conventional:
                message = self.prompt_for_conventional()
            else:
                message = self.prompt_for_message()
        
        # Validate message
        if not self.validate_commit_message(message):
            sys.exit(1)
        
        if conventional and not self.validate_conventional_commit(message):
            sys.exit(1)

        # Commit and push
        self.stage_changes()
        self.commit(message)
        self.push()

        # Success message
        print(f"\n{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
        print(f"{Colors.GREEN}✨ Commit and push completed successfully!{Colors.RESET}")
        print(f"{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Auto git commit and push tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s "feature: add new functionality"
  %(prog)s -i
  %(prog)s -c  (conventional commit mode)
        '''
    )
    
    parser.add_argument(
        'message',
        nargs='?',
        help='Commit message (if not provided, will prompt)'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode (always prompts for message)'
    )
    parser.add_argument(
        '-c', '--conventional',
        action='store_true',
        help='Use conventional commit format'
    )
    
    args = parser.parse_args()
    
    tool = GitAutoCommit()
    tool.run(
        message=args.message,
        interactive=args.interactive,
        conventional=args.conventional
    )


if __name__ == '__main__':
    main()
