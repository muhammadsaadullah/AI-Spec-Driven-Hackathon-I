const fs = require('fs');
const path = require('path');

/**
 * Search API for the Physical AI & Humanoid Robotics course
 * This API serves search queries against the RAG data
 */

class SearchAPI {
  constructor() {
    // Load the RAG data
    this.chunks = [];
    this.documents = [];
    this.loadRAGData();
  }

  loadRAGData() {
    try {
      // Load chunks data
      const chunksPath = path.join(__dirname, 'docs', 'rag-data', 'chunks.json');
      if (fs.existsSync(chunksPath)) {
        this.chunks = JSON.parse(fs.readFileSync(chunksPath, 'utf8'));
      }

      // Load documents data
      const docsPath = path.join(__dirname, 'docs', 'rag-data', 'documents.json');
      if (fs.existsSync(docsPath)) {
        this.documents = JSON.parse(fs.readFileSync(docsPath, 'utf8'));
      }

      console.log(`Loaded ${this.chunks.length} content chunks and ${this.documents.length} documents`);
    } catch (error) {
      console.error('Error loading RAG data:', error.message);
    }
  }

  /**
   * Perform search across the RAG data
   * @param {string} query - The search query
   * @param {number} limit - Maximum number of results to return
   * @returns {Array} - Array of search results
   */
  search(query, limit = 10) {
    if (!query || !query.trim()) {
      return [];
    }

    const searchTerm = query.toLowerCase().trim();
    const terms = searchTerm.split(/\s+/);

    // Calculate relevance score for each chunk
    const scoredResults = this.chunks.map(chunk => {
      let score = 0;
      const contentLower = chunk.content.toLowerCase();
      const titleLower = chunk.title.toLowerCase();

      // Score based on term matches in content and title
      terms.forEach(term => {
        // Title matches are worth more
        if (titleLower.includes(term)) {
          score += 10;
        }
        // Content matches
        if (contentLower.includes(term)) {
          // Count occurrences for additional scoring
          const matches = (contentLower.match(new RegExp(term, 'g')) || []).length;
          score += matches * 2;
        }
      });

      // Additional scoring factors
      // Boost if query appears in the first part of content
      if (contentLower.indexOf(searchTerm) >= 0 && contentLower.indexOf(searchTerm) < 200) {
        score += 5;
      }

      // Boost if query appears in metadata
      if (chunk.metadata && chunk.metadata.keywords) {
        chunk.metadata.keywords.forEach(keyword => {
          if (keyword.toLowerCase().includes(searchTerm)) {
            score += 5;
          }
        });
      }

      return {
        id: chunk.id,
        title: chunk.title,
        url: chunk.url,
        snippet: chunk.content.substring(0, 200) + '...', // First 200 chars as snippet
        score: score,
        metadata: chunk.metadata
      };
    });

    // Filter out results with no score and sort by score
    const results = scoredResults
      .filter(result => result.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return results;
  }

  /**
   * Get search results with proper formatting for the API contract
   */
  getSearchResults(query, limit = 10) {
    const results = this.search(query, limit);
    return {
      results: results,
      total: results.length
    };
  }
}

// Create and export the search API instance
const searchAPI = new SearchAPI();

module.exports = searchAPI;

// If running as a standalone script, start a simple server
if (require.main === module) {
  const http = require('http');
  const url = require('url');

  const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const path = parsedUrl.pathname;
    const query = parsedUrl.query;

    // Set CORS headers for cross-origin requests
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Requested-With');

    // Handle preflight requests
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    if (path === '/search' && req.method === 'GET') {
      const searchQuery = query.q || '';
      const limit = parseInt(query.limit) || 10;

      if (!searchQuery) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Query parameter "q" is required' }));
        return;
      }

      const results = searchAPI.getSearchResults(searchQuery, limit);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(results));
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Endpoint not found' }));
    }
  });

  const PORT = process.env.PORT || 5000;
  server.listen(PORT, () => {
    console.log(`Search API server running on port ${PORT}`);
    console.log(`Example: http://localhost:${PORT}/search?q=robotics&limit=5`);
  });
}