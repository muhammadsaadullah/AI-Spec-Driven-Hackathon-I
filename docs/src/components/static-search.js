
// Client-side search functionality for static deployment
class StaticSearch {
  constructor() {
    this.searchIndex = [];
    this.isLoaded = false;
  }

  async loadSearchIndex() {
    try {
      const response = await fetch('/static-search-index.json');
      this.searchIndex = await response.json();
      this.isLoaded = true;
      console.log('Search index loaded with', this.searchIndex.length, 'entries');
    } catch (error) {
      console.error('Failed to load search index:', error);
      this.isLoaded = false;
    }
  }

  search(query, limit = 10) {
    if (!this.isLoaded || !query || !query.trim()) {
      return [];
    }

    const searchTerm = query.toLowerCase().trim();
    const terms = searchTerm.split(/s+/);

    // Calculate relevance score for each entry
    const scoredResults = this.searchIndex.map(entry => {
      let score = 0;
      const contentLower = entry.content.toLowerCase();
      const titleLower = entry.title.toLowerCase();

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

        // Keyword matches
        if (entry.metadata?.keywords) {
          entry.metadata.keywords.forEach(keyword => {
            if (keyword.toLowerCase().includes(term)) {
              score += 5;
            }
          });
        }
      });

      // Boost if query appears in the first part of content
      if (contentLower.indexOf(searchTerm) >= 0 && contentLower.indexOf(searchTerm) < 200) {
        score += 5;
      }

      return {
        ...entry,
        score: score
      };
    });

    // Filter out results with no score and sort by score
    const results = scoredResults
      .filter(result => result.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return results;
  }
}

// Global instance
const staticSearch = new StaticSearch();

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
  staticSearch.loadSearchIndex();
});
