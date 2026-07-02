# Contributing to Heart Disease Analysis System

Thank you for your interest in contributing to this project!

## Project Overview

This is an AI-assisted cardiac ultrasound (TTE) analysis system implementing:
- U-Net deep learning segmentation
- Clinical decision support
- Clinician review workflow
- HIPAA-compliant data handling
- Multi-site calibration

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/heart-disease-analysis.git
   cd heart-disease-analysis
   ```

3. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Follow PEP 8
- Use meaningful variable names
- Include docstrings for all functions
- Add comments for non-obvious logic

### Testing
- Write tests for new features
- Run existing tests before submitting PR
- Aim for >80% code coverage

### Documentation
- Update README.md if adding features
- Document API changes
- Include usage examples
- Add references to academic papers where applicable

### Commits
- Write clear, descriptive commit messages
- Reference issues in commits: `Fixes #123`
- Keep commits focused on single features

Example:
```
feat: Add multi-view segmentation support

- Implement PLAX view handling
- Add view-specific bias calibration
- Update validation metrics

Fixes #45
```

## Types of Contributions

### Bug Reports
- Use GitHub Issues
- Include reproduction steps
- Provide system information (OS, Python version, GPU, etc.)

### Feature Requests
- Describe the use case
- Explain why it's needed
- Suggest implementation approach

### Code Contributions
1. Address an open issue
2. Follow development guidelines
3. Submit pull request with:
   - Clear description
   - Linked issue
   - Test results
   - Updated documentation

### Documentation
- Fix typos
- Clarify confusing sections
- Add examples
- Translate to other languages

## Architecture & Design Principles

Before contributing, understand:

1. **Clinical Precedence Rule**: AI findings require clinician sign-off
2. **Uncertainty Quantification**: Report error bounds, not point estimates
3. **Regulatory Compliance**: HIPAA, regional approval, audit trails
4. **Per-Site Calibration**: Different equipment needs adjustment
5. **Interpretability**: Feature extraction via deterministic algorithms

See README.md Section 2 for full architecture.

## Testing

```bash
# Run tests
pytest tests/

# Check coverage
pytest --cov=heart_valve_analysis tests/

# Code style
black .
flake8 .
mypy .
```

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review from maintainers
5. Address feedback
6. Merge when approved

## Legal

By contributing, you agree to license your contributions under the MIT License.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow medical ethics guidelines
- Respect patient privacy and data security

## Questions?

- Open a GitHub Discussion
- Check existing issues
- Review documentation in START_HERE.md

Thank you for contributing! 🏥❤️
