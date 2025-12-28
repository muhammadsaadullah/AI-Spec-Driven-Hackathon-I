import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import { useSearchPage } from '@docusaurus/theme-common/internal';
import styles from './styles.module.css';

const CustomSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);

  // Search function that tries API first, then falls back to static search
  const performSearch = async (searchQuery) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    // Show loading state
    setResults([{ title: 'Searching...', url: '#', excerpt: 'Please wait...' }]);

    // First, try the API search (for development or when API is available)
    try {
      // Determine the API endpoint based on environment
      const isDev = process.env.NODE_ENV === 'development';
      const apiEndpoint = isDev
        ? 'http://localhost:5000/search'
        : `${window.location.protocol}//${window.location.hostname}:5000/search`;

      // Call the search API - using the API contract from api-contract.yml
      const response = await fetch(`${apiEndpoint}?q=${encodeURIComponent(searchQuery)}&limit=10`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();

        if (data.results && data.results.length > 0) {
          // Format results to match what the component expects
          const formattedResults = data.results.map(result => ({
            title: result.title,
            url: result.url,
            excerpt: result.snippet || result.content
          }));

          setResults(formattedResults);
          return; // Success with API search
        } else {
          // No results from API, continue to static search
          setResults([]);
        }
      } else {
        // API returned error, continue to static search
        setResults([]);
      }
    } catch (error) {
      console.log('API search failed, falling back to static search:', error.message);
      // Continue to static search
    }

    // If API search failed or no results, use static client-side search
    try {
      // Load the static search index
      const response = await fetch('/static-search-index.json');
      const searchIndex = await response.json();

      // Perform client-side search
      const searchTerm = searchQuery.toLowerCase().trim();
      const terms = searchTerm.split(/\s+/);

      // Calculate relevance score for each entry
      const scoredResults = searchIndex.map(entry => {
        let score = 0;
        const contentLower = entry.content.toLowerCase();
        const titleLower = entry.title.toLowerCase();

        // Score based on term matches in content and title
        terms.forEach(term => {
          // Title matches are worth more
          if (titleLower.includes(term)) {
            score += 15; // Increased weight for titles
          }
          // Content matches
          if (contentLower.includes(term)) {
            // Count occurrences for additional scoring
            const matches = (contentLower.match(new RegExp(term, 'g')) || []).length;
            score += matches * 3; // Increased weight for content matches
          }

          // Keyword matches
          if (entry.metadata?.keywords) {
            entry.metadata.keywords.forEach(keyword => {
              if (keyword.toLowerCase().includes(term)) {
                score += 8; // Increased weight for keyword matches
              }
            });
          }
        });

        // Boost if query appears in the first part of content
        if (contentLower.indexOf(searchTerm) >= 0 && contentLower.indexOf(searchTerm) < 200) {
          score += 10; // Increased boost for early content matches
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
        .slice(0, 10); // Limit to 10 results

      // Format results to match what the component expects
      const formattedResults = results.map(result => ({
        title: result.title,
        url: result.url,
        excerpt: result.content.length > 200 ? result.content.substring(0, 200) + '...' : result.content
      }));

      setResults(formattedResults);
    } catch (error) {
      console.error('Static search error:', error);
      setResults([]); // No results if both methods fail
    }
  };

  useEffect(() => {
    if (query) {
      performSearch(query);
      setShowResults(true);
    } else {
      setResults([]);
      setShowResults(false);
    }
  }, [query]);

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  return (
    <div className={styles.searchContainer}>
      <div className={styles.searchWrapper}>
        <input
          type="text"
          placeholder="Search..."
          value={query}
          onChange={handleInputChange}
          onFocus={() => query && setShowResults(true)}
          className={styles.searchInput}
        />
        <button className={styles.searchButton}>
          🔍
        </button>
      </div>

      {showResults && results.length > 0 && (
        <div className={styles.searchResults}>
          {results.map((result, index) => {
            // Check if this is the loading indicator
            if (result.title === 'Searching...') {
              return (
                <div key={index} className={styles.loading}>
                  <p>Searching...</p>
                </div>
              );
            }
            return (
              <Link
                key={index}
                to={result.url}
                className={styles.searchResultItem}
                onClick={() => {
                  setShowResults(false);
                  setQuery('');
                }}
              >
                <h4>{result.title}</h4>
                <p>{result.excerpt}</p>
              </Link>
            );
          })}
        </div>
      )}

      {showResults && results.length === 0 && query && results[0]?.title !== 'Searching...' && (
        <div className={styles.searchResults}>
          <p className={styles.noResults}>No results found for "{query}"</p>
        </div>
      )}
    </div>
  );
};

export default CustomSearch;