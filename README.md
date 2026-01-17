# Neuracore Types

Shared type definitions for the Neuracore platform. This package maintains a single source of truth for data types in Python (Pydantic models) and automatically generates TypeScript types.

## Overview

- **Python Package**: `neuracore-types` - Pydantic models for Python backend
- **NPM Package**: `@neuracore/types` - TypeScript types for frontend

## Installation

### Python

```bash
pip install neuracore-types
```

### TypeScript/JavaScript

```bash
npm install @neuracore/types
# or
yarn add @neuracore/types
# or
pnpm add @neuracore/types
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/neuracoreai/neuracore_types.git
cd neuracore_types

# Install Python dependencies
pip install -e ".[dev]"

# Install Node dependencies
npm install
```

### Generate TypeScript Types

The TypeScript types are automatically generated from the Python Pydantic models:

```bash
npm install json-schema-to-typescript
python scripts/generate_types.py
```

This will:
1. Read the Pydantic models from `neuracore_types/neuracore_types.py`
2. Generate TypeScript definitions in `typescript/neuracore_types.ts`
3. Create an index file at `typescript/index.ts`

### Build TypeScript Package

```bash
npm run build
```

This compiles the TypeScript files to JavaScript and generates type declarations in the `dist/` directory.

## Release Process

### Creating PRs

All PRs must follow these conventions:

1. **Version Label**: Add exactly one version label to your PR:
   - `version:major` - Breaking changes
   - `version:minor` - New features
   - `version:patch` - Bug fixes
   - `version:none` - No release (docs, chores, etc.)

2. **Commit Format**: PR title and all commits must use conventional commit format:
   ```
   <prefix>: <description>
   ```
   Valid prefixes: `feat`, `fix`, `chore`, `docs`, `ci`, `test`, `refactor`, `style`, `perf`

   Examples:
   - `feat: add new data type for robot state`
   - `fix: resolve serialization issue in TypeScript types`
   - `chore: update dependencies`

### Pending Changelog

For significant changes (`version:major` or `version:minor`), update `changelogs/pending-changelog.md`:

```markdown
## Summary

This release adds support for new sensor data types and improves TypeScript type generation.
```

Simply append your summary to the existing content. This will appear at the top of the release notes.

### Triggering a Release

Releases are manual and triggered via GitHub Actions:

1. Go to **Actions** → **Release** → **Run workflow**
2. Optional: Check **dry_run** to preview without publishing
3. The workflow will:
   - Analyze all PRs since last release
   - Determine version bump (highest priority across all PRs)
   - Generate changelog with all PRs grouped by type
   - Bump version in `pyproject.toml`, `package.json`, and `__init__.py`
   - Generate TypeScript types from Python models
   - Publish Python package to PyPI
   - Build and publish npm package to npm registry
   - Create GitHub release

**Dry run** shows what would happen without making any changes - useful for testing before a real release.

## CI/CD

The repository includes GitHub Actions workflows:

1. **PR Checks**:
   - Validates version labels
   - Enforces conventional commit format
   - Runs pre-commit hooks
   - Suggests changelog updates for major/minor changes

2. **Release** (manual trigger):
   - Generates TypeScript types from Python models
   - Builds and validates both packages
   - Publishes to PyPI and npm registry
   - Creates GitHub release with changelog
