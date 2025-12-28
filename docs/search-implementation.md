# Local Search Implementation for Physical AI & Humanoid Robotics Course

## Overview

This document describes the local search functionality implemented for the course documentation. The search system uses a Retrieval Augmented Generation (RAG) approach to provide relevant search results from the course content.

## Architecture

### Components

1. **Content Extraction System** (`docs/scripts/rag-content-extractor.js`)
   - Extracts content from all markdown files in the docs directory
   - Splits content into overlapping chunks for better context retrieval
   - Stores extracted content in JSON format

2. **Search API** (`search-api.js`)
   - Node.js server that serves search queries
   - Loads RAG data from `docs/rag-data/` directory
   - Implements relevance scoring algorithm
   - Provides REST API endpoint at `/search`

3. **Frontend Component** (`docs/src/components/CustomSearch.js`)
   - React component that provides search UI
   - Connects to the search API
   - Displays formatted search results

### Data Flow

1. Content is extracted from markdown files using the RAG extractor
2. Chunks are stored in `docs/rag-data/chunks.json` and `docs/rag-data/documents.json`
3. Search API loads the RAG data at startup
4. Frontend component calls the API when users enter search queries
5. API returns relevant results based on keyword matching and scoring

## API Endpoints

### GET /search

Searches through course content.

**Parameters:**
- `q` (required): Search query string
- `limit` (optional): Maximum number of results to return (default: 10)

**Response:**
```json
{
  "results": [
    {
      "id": "chunk_id",
      "title": "Result title",
      "url": "/path/to/content",
      "snippet": "Brief content snippet...",
      "score": 15.5
    }
  ],
  "total": 5
}
```

## Scoring Algorithm

The search uses a multi-factor scoring system:
- Term matches in title (10 points per match)
- Term matches in content (2 points per occurrence)
- Exact phrase matches in first 200 characters (5 bonus points)
- Keyword matches in metadata (5 bonus points per match)

## Setup and Running

1. Extract content (if not already done):
   ```bash
   node docs/scripts/rag-content-extractor.js
   ```

2. Start the search API:
   ```bash
   node search-api.js
   ```

3. The API will be available at `http://localhost:5000`

## Integration with Docusaurus

The CustomSearch component is designed to work with Docusaurus and makes requests to the external search API. When deployed, you'll need to ensure the search API is accessible from the frontend.

## Future Enhancements

- Implement vector similarity search for semantic understanding
- Add search result caching for improved performance
- Implement search analytics to improve relevance
- Add faceted search capabilities