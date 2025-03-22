# Contributing to roll-to-train

Thank you for your interest in contributing to **roll-to-train**! This guide will help you understand how to contribute effectively.

## Reporting Issues

1. Check if your issue already exists in [Issues](https://github.com/sbartlett97/roll-to-train/issues)
2. If not, create a new issue including:
   - Clear title
   - Steps to reproduce (for bugs)
   - Environment details (OS, Python version, etc.)
   - Expected behavior

## Contributing Code

### Setup
```bash
# Fork the repository first, then:
git clone https://github.com/sbartlett97/roll-to-train.git
cd roll-to-train
```

### Development Flow
1. Create a descriptive branch:
   ```bash
   git checkout -b fix-issue-123
   # or
   git checkout -b feature-new-functionality
   ```

2. Make your changes:
   - Keep changes focused and modular
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

3. Test your changes:
   ```bash
   pytest
   ```

4. Commit and push:
   ```bash
   git add .
   git commit -m "fix: critical bug in d20 scaling logic (Fixes #123)"
   git push origin your-branch-name
   ```

5. Open a Pull Request:
   - Provide clear description
   - Link related issues
   - Ensure all tests pass

## Development Standards

### Code Quality
- Follow PEP 8 style guide
- Include type hints and docstrings
- Maintain GPU efficiency
- Write comprehensive tests

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
- `fix:` Bug fixes
- `feat:` New features
- `docs:` Documentation changes
- `style:` Code style changes
- `test:` Test changes
- `refactor:` Code refactoring

## Need Help?
- Check the [README.md](README.md)
- Search [Issues](https://github.com/sbartlett97/roll-to-train/issues)
- Create a new issue for questions

Thank you for helping improve roll-to-train!

