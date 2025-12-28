# Search Functionality Deployment Guide

## Overview

The Physical AI & Humanoid Robotics Course includes an advanced search functionality that works in two modes:
1. **API Mode**: Uses a separate Node.js server for search (recommended for development)
2. **Static Mode**: Uses client-side search with pre-generated index (recommended for static deployment like GitHub Pages)

## Development Setup

For development with full search capabilities:

1. Start the search API server:
   ```bash
   npm run search-api
   ```
   This starts the search API on `http://localhost:5000`

2. In another terminal, start the Docusaurus development server:
   ```bash
   cd docs
   npm run start
   ```

3. The search will automatically use the API when available, falling back to static search when the API is not reachable.

## Production Deployment

For deployment to GitHub Pages or other static hosting:

1. The build process automatically generates the static search index:
   ```bash
   npm run docs:build
   ```
   This runs `generate-search-index` before building, creating `static-search-index.json`

2. The search component will automatically:
   - Try to use the API server if available
   - Fall back to the static search index if the API is not accessible

## Architecture

- `search-api.js`: Node.js server that provides search API endpoint
- `generate-static-search.js`: Script that creates static search index for client-side search
- `docs/static-search-index.json`: Generated search index for static deployments
- `docs/src/components/CustomSearch.js`: React component with hybrid search logic

## API Endpoint

The search API provides the following endpoint:
- `GET /search?q={query}&limit={limit}` - Search course content

## Search Index Generation

The search index is created from the RAG data in `docs/rag-data/` and includes:
- Content chunks with relevance scoring
- Metadata and keywords
- Document titles and URLs

## Configuration

The search component automatically detects the environment:
- In development: tries API at `http://localhost:5000/search`
- In production: tries API at `http://[current-domain]:5000/search`, falls back to static search