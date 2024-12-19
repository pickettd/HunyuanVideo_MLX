# Contributing to HunyuanVideo MLX

Thank you for your interest in contributing to HunyuanVideo MLX! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/HunyuanVideo_MLX.git
cd HunyuanVideo_MLX
```

2. Install development dependencies:
```bash
./install_mlx.sh --with-dev
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Keep functions focused and modular
- Add docstrings for public functions and classes

### MLX-Specific Guidelines
- Use MLX operations instead of NumPy where possible
- Clear MLX cache regularly in memory-intensive operations
- Use fp16 precision for Metal optimization
- Handle tensor device placement explicitly

### Testing
- Add tests for new features
- Run tests before submitting PR:
```bash
pytest tests/
```

### Commits
- Use clear, descriptive commit messages
- Keep commits focused and atomic
- Reference issues in commit messages when applicable

## Pull Request Process

1. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "feat: add your feature description"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Create a Pull Request with:
   - Clear description of changes
   - Any relevant issue references
   - Screenshots/videos if applicable
   - List of tested Mac models/configurations

## Issue Guidelines

When creating issues:
- Use issue templates when available
- Include your system configuration
- Provide clear reproduction steps
- Include relevant logs/error messages
- Specify MLX and macOS versions

## License

By contributing, you agree that your contributions will be licensed under the project's Tencent Hunyuan Community License.
