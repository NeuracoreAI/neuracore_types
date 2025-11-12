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

## CI/CD

The repository includes GitHub Actions workflows that:

1. **On every push to `main` or PR**:
   - Automatically generates TypeScript types from Python models
   - Builds and validates both packages
   - Publishes Python package to PyPI
   - Publishes NPM package to npm registry
