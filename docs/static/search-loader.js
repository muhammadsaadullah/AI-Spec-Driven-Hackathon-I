// Search loader script that runs after page load
(function() {
  // Wait for the page to fully load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSearch);
  } else {
    setTimeout(initializeSearch, 500); // Small delay to ensure DOM is ready
  }

  function initializeSearch() {
    // Wait a bit more to ensure navbar is fully rendered
    setTimeout(function() {
      const searchContainer = document.getElementById('custom-search-container');
      if (searchContainer && searchContainer.children.length === 0) {
        // Create search elements
        const searchDiv = document.createElement('div');
        searchDiv.style.position = 'relative';
        searchDiv.style.display = 'inline-block';
        searchDiv.style.marginLeft = '1rem';
        searchDiv.style.minWidth = '200px';

        const searchWrapper = document.createElement('div');
        searchWrapper.style.display = 'flex';
        searchWrapper.style.alignItems = 'center';
        searchWrapper.style.border = '1px solid #d0d5dd';
        searchWrapper.style.borderRadius = '6px';
        searchWrapper.style.background = 'white';
        searchWrapper.style.boxShadow = '0 1px 2px rgba(16, 24, 40, 0.05)';
        searchWrapper.style.transition = 'box-shadow 0.2s ease, border-color 0.2s ease';

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search...';
        searchInput.style.padding = '10px 12px';
        searchInput.style.border = 'none';
        searchInput.style.borderRadius = '6px 0 0 6px';
        searchInput.style.outline = 'none';
        searchInput.style.width = '150px';
        searchInput.style.fontSize = '14px';
        searchInput.style.flexGrow = '1';

        const searchButton = document.createElement('button');
        searchButton.innerHTML = '🔍';
        searchButton.style.background = '#25c2a0';
        searchButton.style.color = 'white';
        searchButton.style.border = 'none';
        searchButton.style.borderRadius = '0 6px 6px 0';
        searchButton.style.padding = '10px 12px';
        searchButton.style.cursor = 'pointer';
        searchButton.style.fontSize = '14px';
        searchButton.style.display = 'flex';
        searchButton.style.alignItems = 'center';
        searchButton.style.justifyContent = 'center';

        searchWrapper.appendChild(searchInput);
        searchWrapper.appendChild(searchButton);
        searchDiv.appendChild(searchWrapper);

        // Add event listeners
        let searchTimeout;
        function performSearch(query) {
          if (!query.trim()) return;

          // Try API first, then fallback to static search
          const isDev = window.location.hostname === 'localhost';
          const apiEndpoint = isDev
            ? 'http://localhost:5000/search'
            : `${window.location.protocol}//${window.location.hostname}:5000/search`;

          // Show loading state
          showSearchResults([{ title: 'Searching...', url: '#', excerpt: 'Please wait...' }]);

          fetch(`${apiEndpoint}?q=${encodeURIComponent(query)}&limit=10`)
            .then(response => {
              if (response.ok) {
                return response.json();
              }
              throw new Error('API request failed');
            })
            .then(data => {
              if (data.results && data.results.length > 0) {
                const results = data.results.map(result => ({
                  title: result.title,
                  url: result.url,
                  excerpt: result.snippet || result.content
                }));
                showSearchResults(results);
              } else {
                // Fallback to static search
                fallbackSearch(query);
              }
            })
            .catch(() => {
              // API failed, use static search
              fallbackSearch(query);
            });
        }

        function fallbackSearch(query) {
          // Load static search index and search locally
          fetch('/static-search-index.json')
            .then(response => response.json())
            .then(index => {
              const searchTerm = query.toLowerCase().trim();
              const terms = searchTerm.split(/\s+/);

              // Calculate relevance score
              const scoredResults = index.map(entry => {
                let score = 0;
                const contentLower = entry.content.toLowerCase();
                const titleLower = entry.title.toLowerCase();

                for (const term of terms) {
                  if (titleLower.includes(term)) score += 15;
                  if (contentLower.includes(term)) {
                    const matches = (contentLower.match(new RegExp(term, 'gi')) || []).length;
                    score += matches * 3;
                  }
                  if (entry.metadata?.keywords) {
                    for (const keyword of entry.metadata.keywords) {
                      if (keyword.toLowerCase().includes(term)) {
                        score += 8;
                      }
                    }
                  }
                }

                if (contentLower.indexOf(searchTerm) >= 0 && contentLower.indexOf(searchTerm) < 200) {
                  score += 10;
                }

                return { ...entry, score };
              });

              const results = scoredResults
                .filter(result => result.score > 0)
                .sort((a, b) => b.score - a.score)
                .slice(0, 10);

              const formattedResults = results.map(result => ({
                title: result.title,
                url: result.url,
                excerpt: result.content.length > 200 ? result.content.substring(0, 200) + '...' : result.content
              }));

              showSearchResults(formattedResults);
            })
            .catch(error => {
              console.error('Static search error:', error);
              showSearchResults([]);
            });
        }

        function showSearchResults(results) {
          // Remove existing results
          const existingResults = searchDiv.querySelector('.search-results');
          if (existingResults) existingResults.remove();

          if (results.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'search-results';
            noResults.style.position = 'absolute';
            noResults.style.top = 'calc(100% + 8px)';
            noResults.style.right = '0';
            noResults.style.background = 'white';
            noResults.style.border = '1px solid #e5e7eb';
            noResults.style.borderRadius = '8px';
            noResults.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';
            noResults.style.width = '400px';
            noResults.style.maxHeight = '400px';
            noResults.style.overflowY = 'auto';
            noResults.style.zIndex = '1000';
            noResults.style.marginTop = '4px';
            noResults.style.backgroundClip = 'padding-box';
            noResults.style.padding = '20px';
            noResults.style.textAlign = 'center';
            noResults.style.color = '#6b7280';
            noResults.style.fontSize = '14px';
            noResults.innerHTML = 'No results found';
            searchDiv.appendChild(noResults);

            setTimeout(() => {
              if (noResults.parentNode) noResults.parentNode.removeChild(noResults);
            }, 3000);
            return;
          }

          const resultsDiv = document.createElement('div');
          resultsDiv.className = 'search-results';
          resultsDiv.style.position = 'absolute';
          resultsDiv.style.top = 'calc(100% + 8px)';
          resultsDiv.style.right = '0';
          resultsDiv.style.background = 'white';
          resultsDiv.style.border = '1px solid #e5e7eb';
          resultsDiv.style.borderRadius = '8px';
          resultsDiv.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)';
          resultsDiv.style.width = '400px';
          resultsDiv.style.maxHeight = '400px';
          resultsDiv.style.overflowY = 'auto';
          resultsDiv.style.zIndex = '1000';
          resultsDiv.style.marginTop = '4px';
          resultsDiv.style.backgroundClip = 'padding-box';

          results.forEach(result => {
            const item = document.createElement('a');
            item.href = result.url;
            item.className = 'search-result-item';
            item.style.display = 'block';
            item.style.padding = '12px 16px';
            item.style.textDecoration = 'none';
            item.style.color = '#333';
            item.style.borderBottom = '1px solid #f3f4f6';
            item.style.transition = 'background-color 0.15s ease';
            item.style.cursor = 'pointer';

            item.onmouseenter = () => item.style.backgroundColor = '#f9fafb';
            item.onmouseleave = () => item.style.backgroundColor = 'white';

            item.onclick = () => {
              // Close results and clear search
              resultsDiv.style.display = 'none';
              searchInput.value = '';
            };

            const title = document.createElement('h4');
            title.style.margin = '0 0 6px 0';
            title.style.color = '#25c2a0';
            title.style.fontSize = '15px';
            title.style.fontWeight = '600';
            title.style.lineHeight = '1.4';
            title.textContent = result.title;

            const excerpt = document.createElement('p');
            excerpt.style.margin = '0';
            excerpt.style.fontSize = '13px';
            excerpt.style.color = '#6b7280';
            excerpt.style.lineHeight = '1.5';
            excerpt.textContent = result.excerpt;

            item.appendChild(title);
            item.appendChild(excerpt);
            resultsDiv.appendChild(item);
          });

          // Add event to close results when clicking outside
          document.addEventListener('click', function closeHandler(event) {
            if (!searchDiv.contains(event.target)) {
              resultsDiv.style.display = 'none';
              document.removeEventListener('click', closeHandler);
            }
          });

          searchDiv.appendChild(resultsDiv);
        }

        searchInput.addEventListener('input', function(e) {
          const query = e.target.value;

          // Clear previous timeout
          if (searchTimeout) {
            clearTimeout(searchTimeout);
          }

          // Set new timeout for debouncing
          if (query.trim()) {
            searchTimeout = setTimeout(() => {
              performSearch(query);
            }, 300);
          } else {
            // Remove results when input is cleared
            const existingResults = searchDiv.querySelector('.search-results');
            if (existingResults) existingResults.remove();
          }
        });

        searchButton.addEventListener('click', function() {
          performSearch(searchInput.value);
        });

        searchInput.addEventListener('keypress', function(e) {
          if (e.key === 'Enter') {
            performSearch(searchInput.value);
          }
        });

        // Add focus/blur effects to wrapper
        searchInput.addEventListener('focus', function() {
          searchWrapper.style.boxShadow = '0 0 0 3px rgba(13, 110, 253, 0.15)';
          searchWrapper.style.borderColor = '#86b7fe';
        });

        searchInput.addEventListener('blur', function() {
          searchWrapper.style.boxShadow = '0 1px 2px rgba(16, 24, 40, 0.05)';
          searchWrapper.style.borderColor = '#d0d5dd';
        });

        searchContainer.appendChild(searchDiv);
      }
    }, 1000);
  }
})();